# Energy Side-Channel MIA

Code for our paper: Side-Channel Privacy Attacks on Energy-Efficient Mobile Agents: A Membership Inference Perspect



## What this is

We show that energy-efficient inference techniques in mobile agents (early-exit, pruning, MoE routing) leak training data membership through power consumption. The attack reads power via standard Android kernel interfaces — no model access, no special permissions needed.

---

## Setup

```bash
git clone https://github.com/yourusername/energy-side-channel-mia
cd energy-side-channel-mia
pip install -r requirements.txt
```

---

## Run

```bash
# Run the attack
python experiments/run_attack.py

# Evaluate defenses
python experiments/run_defenses.py
```

---

## Structure

```
src/
  trigger_design.py          # Poisoning and trigger crafting
  energy_extraction.py       # Power reading from kernel interfaces
  membership_inference.py    # MIA classifier
  defenses/
    dp_sgd.py
    noise_injection.py
    stochastic_early_exit.py
    randomized_pruning.py
  utils.py
experiments/
  run_attack.py
  run_defenses.py
```

---

## Contact

ibtisamehsan146@gmail.com
