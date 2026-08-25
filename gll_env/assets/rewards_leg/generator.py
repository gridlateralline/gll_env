# Copyright 2026 ewz - Zurich Municipal Electric Utility.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Generate the bundled LEG tariff assets.

Every rate here is written in **Rp./kWh** (Swiss centimes) exactly as ewz
publishes them, and converted to CHF/kWh once, in :func:`_daily_rates`.
Consumption rates are negative (a cost to the member), injection rates
positive (a payment to the member); all are quoted *without* taxes, following
the "Nettoprinzip 2022".

Each tariff is a day-long curve over ``NUM_STEPS_PER_DAY`` 15-minute
intervals, split into a peak (HT) and an off-peak (NT) band.

Unlike the grid assets, nothing here needs pandapower — it is pure JAX — but
it is kept out of the import path all the same, so the shipped
``.safetensors`` stay the single source of truth at runtime.

Regenerate with::

    uv run -m gll_env.assets.rewards_leg.generator
"""

from pathlib import Path

import jax.numpy as jnp

from gll_env.assets.serialization import save_asset_arrays

NUM_STEPS_PER_DAY = 24 * 4
START_HT = 6 * 4  # 06:00 -> 06:15 is the first HT interval (inclusive)
END_HT = 22 * 4  # 22:00 -> 22:15 is the last HT interval (exclusive)

RP_PER_CHF = 100.0

# Package path to the bundled safetensors tariff assets.
LEG_ASSETS_DIR = Path(__file__).parent

# --- ewz published components, Rp./kWh, tariff year 2026 ---------------------
# EEA-Tarif (feed-in): energy + HKN + solar subsidy.
# https://www.ewz.ch/de/private/solaranlagen/verrechnungsloesungen/stromruecklieferung.html
EEA_INJECTION_HT = 8.5 + 3 + 2
EEA_INJECTION_NT = 4.45 + 3 + 2

# ewz.natur (default supply): energy + public duties, grid usage added separately.
NATUR_ENERGY_HT = -9.3
NATUR_ENERGY_NT = -4.9
PUBLIC_DUTIES = -5.03

# Grid usage fee, billed to consumption.
GRID_USAGE_HT = -11.92
GRID_USAGE_NT = -5.97

# Share of the grid usage fee waived inside a LEG.
GRID_REBATE_SHARE = 0.4


def _daily_rates(rate_nt_rp: float, rate_ht_rp: float) -> jnp.ndarray:
    """Build a day-long CHF/kWh curve from an off-peak and a peak rate in Rp./kWh.

    Args:
        rate_nt_rp: Off-peak (NT) rate, applied outside ``[START_HT, END_HT)``.
        rate_ht_rp: Peak (HT) rate, applied during ``[START_HT, END_HT)``.

    Returns:
        Shape ``(NUM_STEPS_PER_DAY,)`` float32, in CHF/kWh.
    """
    rates_rp = jnp.full((NUM_STEPS_PER_DAY,), rate_nt_rp, dtype=jnp.float32)
    rates_rp = rates_rp.at[START_HT:END_HT].set(rate_ht_rp)
    return rates_rp / RP_PER_CHF


def _payments(
    leg_injection: tuple[float, float],
    vnb_injection: tuple[float, float],
    leg_consumption: tuple[float, float],
    vnb_consumption: tuple[float, float],
) -> dict[str, jnp.ndarray]:
    """Assemble the four rate curves a :class:`~gll_env.rewards.leg.Payments` needs.

    Each argument is an ``(nt, ht)`` pair in Rp./kWh.
    """
    return {
        "payment_leg_injection": _daily_rates(*leg_injection),
        "payment_vnb_injection": _daily_rates(*vnb_injection),
        "payment_leg_consumption": _daily_rates(*leg_consumption),
        "payment_vnb_consumption": _daily_rates(*vnb_consumption),
    }


def _split_rebate_tariff(
    vnb_injection: tuple[float, float],
    vnb_energy: tuple[float, float],
    grid_usage: tuple[float, float],
    rebate_share: float = GRID_REBATE_SHARE,
) -> dict[str, jnp.ndarray]:
    """A LEG tariff that splits the waived grid usage fee evenly between both sides.

    The VNB rates are the members' fallback: the published feed-in tariff for
    injection, and energy + grid usage + public duties for consumption. Trading
    inside the LEG waives ``rebate_share`` of the grid usage fee, and that saving
    is shared 50/50 — the injector is paid half of it on top of the feed-in
    tariff, the consumer keeps the other half off their bill. Symmetric by
    construction, hence "fair": neither side captures the community's saving.

    Each argument is an ``(nt, ht)`` pair in Rp./kWh; ``grid_usage`` is negative.
    """
    vnb_consumption = tuple(
        energy + usage + PUBLIC_DUTIES for energy, usage in zip(vnb_energy, grid_usage, strict=True)
    )
    # grid_usage is negative, so negate to get the (positive) saving, then halve it.
    rebate_half = tuple(-usage * rebate_share / 2.0 for usage in grid_usage)

    return _payments(
        leg_injection=tuple(
            rate + half for rate, half in zip(vnb_injection, rebate_half, strict=True)
        ),
        vnb_injection=vnb_injection,
        leg_consumption=tuple(
            rate + half for rate, half in zip(vnb_consumption, rebate_half, strict=True)
        ),
        vnb_consumption=vnb_consumption,
    )


def generate_and_save_solarquartier_asset() -> None:
    """ewz Solarquartier-Tarif 2026 — the published product.

    Unlike the constructed tariffs below, this one is not derived from a
    rebate split: ewz quotes a flat LEG feed-in rate and grants the consumer
    the full 40% grid usage rebate, against a higher LEG energy rate (13 Rp.)
    than ewz.natur charges.

    https://www.ewz.ch/de/geschaeftskunden/solarenergie/verrechnungsloesungen/leg-lokale-elektrizitaetsgemeinschaft.html
    """
    leg_energy_ht = -13.0
    leg_energy_nt = -13.0
    leg_grid_usage_ht = GRID_USAGE_HT * (1.0 - GRID_REBATE_SHARE)
    leg_grid_usage_nt = GRID_USAGE_NT * (1.0 - GRID_REBATE_SHARE)

    # LEG feed-in: energy + solar subsidy, flat across HT and NT.
    leg_injection = 12.0 + 2.0

    payments = _payments(
        leg_injection=(leg_injection, leg_injection),
        vnb_injection=(EEA_INJECTION_NT, EEA_INJECTION_HT),
        leg_consumption=(
            leg_energy_nt + leg_grid_usage_nt + PUBLIC_DUTIES,
            leg_energy_ht + leg_grid_usage_ht + PUBLIC_DUTIES,
        ),
        vnb_consumption=(
            NATUR_ENERGY_NT + GRID_USAGE_NT + PUBLIC_DUTIES,
            NATUR_ENERGY_HT + GRID_USAGE_HT + PUBLIC_DUTIES,
        ),
    )
    save_asset_arrays("solarquartier", asset_dir=LEG_ASSETS_DIR, **payments)


def generate_and_save_fair_leg_asset() -> None:
    """ewz's 2026 tariff components, with the grid usage rebate split 50/50."""
    payments = _split_rebate_tariff(
        vnb_injection=(EEA_INJECTION_NT, EEA_INJECTION_HT),
        vnb_energy=(NATUR_ENERGY_NT, NATUR_ENERGY_HT),
        grid_usage=(GRID_USAGE_NT, GRID_USAGE_HT),
    )
    save_asset_arrays("fair_leg", asset_dir=LEG_ASSETS_DIR, **payments)


def generate_and_save_flat_leg_asset() -> None:
    """A synthetic control tariff: same rebate split, but no HT/NT structure.

    Every component is flat across the day, so any behaviour an agent learns
    under it cannot come from time-of-use arbitrage — only from matching
    supply and demand inside the community.
    """
    flat_injection = 9.0
    flat_energy = -9.0
    flat_grid_usage = -9.0

    payments = _split_rebate_tariff(
        vnb_injection=(flat_injection, flat_injection),
        vnb_energy=(flat_energy, flat_energy),
        grid_usage=(flat_grid_usage, flat_grid_usage),
    )
    save_asset_arrays("flat_leg", asset_dir=LEG_ASSETS_DIR, **payments)


if __name__ == "__main__":
    generate_and_save_solarquartier_asset()
    generate_and_save_fair_leg_asset()
    generate_and_save_flat_leg_asset()
