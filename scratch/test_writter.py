# Copyright (C) 2025-2025 The MegaMek Team. All Rights Reserved.
#
# This file is part of MM-Caspar-Trainer.
#
# MM-Caspar-Trainer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License (GPL),
# version 3 or (at your option) any later version,
# as published by the Free Software Foundation.
#
# MM-Caspar-Trainer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# A copy of the GPL should have been included with this project;
# if not, see <https://www.gnu.org/licenses/>.
#
# NOTICE: The MegaMek organization is a non-profit group of volunteers
# creating free software for the BattleTech community.
#
# MechWarrior, BattleMech, `Mech and AeroTech are registered trademarks
# of The Topps Company, Inc. All Rights Reserved.
#
# Catalyst Game Labs and the Catalyst Game Labs logo are trademarks of
# InMediaRes Productions, LLC.

with open("hyper_parameter_search.sh", "w") as f:
    dropout = 0.01
    f.write('''
#!/bin/bash
# Hyper parameter saerch
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=1
export AWS_ACCESS_KEY_ID=mlops-demo
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_SECRET_ACCESS_KEY=mlops-demo
export MLFLOW_TRACKING_URI=http://localhost:9002

# Set the number of intra-op and inter-op threads for TensorFlow
# This helps with CPU parallelization
# Recommended to set these to the number of physical cores
export TF_NUM_INTRAOP_THREADS=12
export TF_NUM_INTEROP_THREADS=12

# Run a small-scale NAS optimized for CPU
echo "Running CPU-optimized Hyperparameter Search..."

''')
    line = 0
    run = 0
    for batch_step in range(5, 250, 10):
        for dropout in range(0, 50, 5):
            a = (f"python -m caspar.__main__ "
              f"--hidden-layers 64 128 32 "
              f"--dropout-rate 015 "
              f"--batch {batch_step} "
              f"--epochs 500 "
              f"--learning-rate 0.1 & \\\n")
            f.write(a)
            line += 1

            if line > 12:
                run += 1
                line = 0
                f.write(f'echo "running batch {run}";\n')
                f.write("wait;\n")

    run += 1
    f.write(f'echo "running batch {run}";\n')
    f.write("wait;\n")

if __name__ == '__main__':
    pass