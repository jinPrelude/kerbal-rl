# Kerbal-rl
kerbal-rl is aerospace reinforcement learning environment using
Kerbal Space Program.

## Environments
Currently available environments are :

- hover-v0 : Given the highest score if the rocket is hovering at the
target altitude. target altitude is given randomly at avery episodes.

- hover-v1 : Same feature with hover-v0, but the sparse is given.

## Installation
<b>To use this package, you must have Kerbal Space Program with krpc
mod installed.</b>


This code will automatically install the package.
```shell
git clone https://github.com/jinPrelude/kerbal-rl.git
cd kerbal-rl
pip install .
```

## Prerequisite

1. Before to run the code, you must have Kerbal Space Program(KSP)
with krpc mod installed.

    - _For more information on Kerbal Space Program please visit
 [here](https://www.kerbalspaceprogram.com/en/)._

    - _For more information on krpc please visit
[here](https://github.com/krpc/krpc)._

2. Run KSP, go to vehicle assembly building, and make your own rocket. If
you are not familiar with making a rocket,
use premade vehicle 'test_vehicle.craft' which is involved in our repo
by following the instruction below :
    1. Copy test_vehicle.craft file.
    2. Move directory to ksp local folder.
    3. Paste the file in Ships\VAB folder.
    4. Run KSP. You could find the 'test_vehicle' in the game.

3. Click the launch button to go to the launch pad.
4. Lastly, Click the button 'Start Server' on kRPC window. If there's no any
windows named kRPC, check our if the krpc mod has successfully installed.


Let's get back to the code. You can simply import & initialize kerbal-rl like :
```shell
import kerbal_rl
env = kerbal_rl.make('hover-v0')
env.reset()
```
You could feel more familiar with kerbal-rl if you
 have experience using Openai gym.

## How it works
kerbal-rl Capture the scene for every 0.1 second for a
step. It means, Although your computing time for desicion making takes
 time more than 0.1 second, ksp doesn't wait for it. Fortunately, There
 is the hyperparameter 'interval', that allows you to adjust so you could
 fine-tuning the most appropriate interval.

```shell
...

env = kerbal_rl.make('hover-v0')

# You can customize your environment like :
env(max_step=100, max_altitude=10, sas=True, interval=0.1)

...
```

