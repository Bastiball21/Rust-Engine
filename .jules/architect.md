## 2024-12-20 - Capacity Upgrade: 512 Hidden Neurons
**Hypothesis:** Doubling the hidden layer size from 256 to 512 will significantly increase the network's capacity to model complex positional nuances, particularly in middlegames, leading to long-term Elo growth.
**Result:** Successfully implemented the topology change in both inference (Rust AVX2) and training (Bullet Graph). Verified graph integrity and gradient flow. Caught a critical "MultipleRoots" bug caused by shadowed variable definitions in the trainer builder.
**Action:** When modifying network graphs, always verify that every defined node is connected to the output path to prevent orphaned roots.
