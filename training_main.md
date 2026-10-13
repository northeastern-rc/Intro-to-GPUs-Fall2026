
<img src="images/NU_logo_white.png" alt="drawing" width="900"/>

<br>
<br>

# Research Computing Training

## Presenter

Ganesan Murugan

Lead Graduate Research Assistant

[Research Computing](https://rc.northeastern.edu/research-computing-team/)

## Introduction to GPUs on Explorer


Welcome to another session in the [Research Computing Fall 2026 Training Series](https://rc.northeastern.edu/training/research-computing-fall-training/)! This guide will help you get started with GPU computing on the Explorer cluster.

This is a beginner-friendly introduction to using the GPUs on the Explorer cluster.

By attending this presentation you will get an introduction to:

1. [What are GPUs, anyway?](#what-are-gpus-anyway)
2. [Accessing Explorer](#accessing-explorer)
3. [Which GPU Partitions are available on Explorer?](#which-gpu-partitions-are-available-on-explorer)
4. [Which GPU Types are available on Explorer?](#which-gpu-types-are-available-on-explorer)
5. [Ok, I know about GPUs on Explorer. How do I use them?](#ok-i-know-about-gpus-on-explorer-how-do-i-use-them)
6. [Prerequisite](#prerequisite)
7. [Open OnDemand (OOD)](#open-ondemand-ood)
8. [Interactive GPU Sessions](#interactive-gpu-sessions)
9. [Batch Jobs with SLURM](#batch-jobs-with-slurm)
10. [Requesting a Specific GPU Type](#requesting-a-specific-gpu-type)
11. [Resource Limits: What Happens When You Hit Them?](#resource-limits-what-happens-when-you-hit-them)
12. [Monitoring Your Jobs](#monitoring-your-jobs)
13. [Using GPUs Efficiently](#using-gpus-efficiently)
14. [Troubleshooting Common Errors](#troubleshooting-common-errors)
15. [Best Practices for Using GPUs on Explorer](#best-practices-for-using-gpus-on-explorer)
16. [How to get help](#how-to-get-help)

All these materials are also available at [Intro To GPUs Github Repo](https://github.com/northeastern-rc/Intro-to-GPUs-Fall2026/blob/main/training_main.md).

You are welcome to follow along by typing the commands in your terminal. You can also just watch the demo, and try it later at your own pace. The recordings for this session will also be made available later on the [Research Computing website](https://rc.northeastern.edu/training/research-computing-fall-training/). 

Let's get started!

## What are GPUs, anyway?

Before we look at that question, can someone share what a core is?
> **Answer:** A core is a processing unit that can independently execute instructions. 

In simple terms:

- CPUs have a few, complex cores.
- GPUs, have thousands of smaller, simpler cores.

<img src="images/cpu_vs_gpu_official.jpg" width="500"/>

(Source: https://blogs.nvidia.com/blog/whats-the-difference-between-a-cpu-and-a-gpu/)


So it depends what do we want to achieve. For tasks that can be massively parallelized, GPUs are the perfect choice.

Consider the analogy of delivering thousands of pizzas across a city. We have a few Ferraris and thousands of bicycles:

- **CPUs (Ferraris):** A few high-speed vehicles that can deliver pizzas quickly to a limited number of locations.
 - **GPUs (Bicycles):** A large fleet of bicycles that can deliver pizzas to many locations in parallel.

Which one would we choose? Fleet of bicycles, of course.

<img src="images/cpu_vs_gpu_pizza_delivery.png" width=500/>

Here is another analogy. Consider a group of professors in a room with large whiteboards, vs. a stadium full of students with handheld calculators and sticky notes. Every student has the data that they need, and they are listening to a megaphone. 

Professors can do complex mathematical operations and problems much faster. 

But if we need to just multiply a very large matrix, meaning computing multiply and add operations for each slot in the resulting matrix, the stadium full of students makes more sense!

The point is, we use GPUs for tasks that can be massively parallelized, such as:
- Machine Learning and Deep Learning
- Scientific Simulations
- Data Analysis and Visualization

### When NOT to use GPUs

A GPU only helps if your code is written to use it. Stick with CPUs (e.g. the `short` partition) when:
- Your code is serial or single-threaded, and does one step at a time.
- Your program doesn't use a GPU-enabled library (PyTorch, TensorFlow, JAX, CuPy, RAPIDS, CUDA code, etc.). Requesting a GPU won't speed it up.
- Your job is mostly file I/O, data cleaning, or small calculations.

Requesting a GPU you don't use means waiting longer in the queue, and keeps that GPU away from someone who needs it.

## Accessing Explorer
To access the Explorer cluster, use SSH to connect from your local machine:
```bash
ssh username@login.explorer.northeastern.edu
```
Replace `username` with your actual username.

## Overview

Cluster (entire system)
  ↓
Partitions (groups of nodes)
  ↓
Nodes (individual servers)
  ↓
Cores + GPUs + Memory (resources within each node)

## Which GPU Partitions are available on Explorer?

Here are 5 (plus one) of the commonly used GPU partitions on Explorer:
- `gpu`: General-purpose GPU partition. (max: 8 hours)
- `gpu-interactive`: GPU sessions for Open OnDemand based development/testing. (max: 2 hours)
- `gpu-short`: Short-duration GPU jobs. Jobs in this partition are prioritized. Use this partition for quick tests or short tasks. (max: 2 hours)
- `multigpu`: For jobs that require multiple GPUs. [Request access here](https://bit.ly/NURC-PartitionAccess) (max: 24 hours)
- `courses-gpu`: Dedicated to courses-related GPU jobs. Instructors are welcome to request access for their courses. (max: 24 hours)
- Bonus: `sharing`: GPUs shared by Northeastern PIs for general community use. (max: 1 hour)

### Partition limits

Each partition has limits. They apply **per user**, counted across all of your jobs:

| Partition | Approval needed? | Time limit (default / max) | Running jobs | Submitted jobs | GPUs per job |
|---|---|---|---|---|---|
| `gpu-short` | No | 1 hour / 2 hours | 2 | 4 | 1 |
| `gpu-interactive` | No | 1 hour / 2 hours | 2 | 4 | 1 |
| `gpu` | No | 4 hours / 8 hours | 4 | 8 | 1 |
| `multigpu` | Yes | 12 hours / 24 hours | 4 | 8 | 8 |
| `sharing` | No | 30 minutes / 60 minutes | 2 | 4 | 1 |

A few things to note:
- If you don't pass `--time`, your job gets the **default** time limit, not the max. Set `--time` yourself if you need longer.
- The `multigpu` partition requires a [Partition Access Request](https://bit.ly/NURC-PartitionAccess). You'll be asked for test results showing that your code really uses more than one GPU.
- Use `scontrol show partition <partition_name>` to see the live configuration of any partition.

For the full table, see the [Partitions documentation](https://rc.northeastern.edu/partitions/).

## Which GPU Types are available on Explorer?

For any given partition, we can type the following command in the Explorer terminal to see the available GPU types on that partition:

```bash
sinfo -p <partition_name> -o "%G"
```
For example, for the `gpu` partition, we have:

```bash
sinfo -p gpu -o "%G"
GRES
gpu:v100-sxm2:4(S:0-1)
gpu:t4:4(S:0-1)
gpu:a100:3
gpu:a100:4
gpu:h200:8
gpu:v100-pcie:2(S:0-1)
gpu:v100-sxm2:3(S:0-1)
```

> 💡 **Question for Audience:** Which GPUs are available in the `sharing` partition? 

Hint: (Try `sinfo -p sharing -o "%G"`)

> 💡 **Question for Audience:** Would it be useful to know which GPUs are available on a partition before submitting a job? 

**Hint:** Try the following command:
```bash
sinfo -p gpu -O "NodeList,Gres:30,GresUsed:30"
```

We also have a GPU-monitor app available on Open OnDemand (OOD) that shows which GPU nodes are available/Idle.

<img src="images/gpu-monitor1.png" width="200"/>

<img src="images/gpu-monitor2.png" width="500"/>

Once a certain GPU type node seems free to you, use the following command in terminal to see if the node is indeed available.

```bash
 sinfo -n nodename --Format="Gres:30,GresUsed:30" 
```

Please note that "MIXED" state does not mean that all GPUs are available. "MIXED" refers to the state of all the "CPUs" on the node, and not the GPUs. So that is why we need to check if the GPUs are available using the above command.


## Ok, I know about GPUs on Explorer. How do I use them?

There are three main ways to use GPUs on Explorer:
1. **Open OnDemand (OOD):** A web-based interface that allows you to run GPU jobs interactively.

2. **Interactive Sessions:** You can request an interactive GPU session using `srun`

3. **Batch Jobs:** You can submit batch jobs to the GPU partitions using SLURM job scripts.

## Prerequisite

Let's clone the directory that we'll be using for this demo

1. In a terminal, let's login to `username@login.explorer.northeastern.edu`
2. After logging in, let's type `cd /scratch/$USER`. This takes us to our own scratch directory.
3. Run the command `git clone git@github.com:northeastern-rc/Intro-to-GPUs-Fall2026.git`

## Open OnDemand (OOD):

We will run `gpu_simple.py` in Open OnDemand (OOD), by following these steps.

1. Let's go to ood.explorer.northeastern.edu. Login.
2. Under `Standard Apps`, let's select VSCode Server

<img src="images/ood_main.png"/>

3. Enter a (simple) password, (e.g. `123`), select `sharing` as partition, and select `1` GPU. Other items can be left as default.
4. Hit Launch. 

<img src="images/vscode_on_ood.png"/>

5. Once the session is connected, open the folder that we just cloned in VSCode.

<img src="images/vscode.png"/>

6. Also open the Terminal by selecting View -> Terminal. Go to the directory we just cloned.
7. Run the command `chmod +x setup_python_env.sh` in the terminal (This needs to be done only once to make it executable) 
8. Run `./setup_python_env.sh` to activate the environment.
9. Let's verify if the right python is selected by typing `which python`. It should point to the virtual environment.
10. Now we are ready to test. In a terminal type `python gpu_simple.py`. It should say `CUDA_VISIBLE_DEVICES = 0`
11. (Optional) For some more items, we can run  `python example1/gpu_test_torch.py`
12. This concludes our demo for using Open OnDemand.

## Interactive GPU Sessions:

Since the environment is setup already, we can simply use this environment for an interactive session.

1. Login to Explorer using a terminal: `ssh username@login.explorer.northeastern.edu`
2. Go to the directory we cloned using the command `cd /scratch/$USER/`
3. To get a GPU based session, type `srun --partition=sharing --gres=gpu:1 --pty bash`
4. We can now activate our environment and have the python script run.
```bash
./setup_python_env.sh
which python
python example1/gpu_test_torch.py
``` 
5. We can also see what is in that file (and potentially modify if we need to). Try `vi example1/gpu_test_torch.py`. Hit `i` and insert an `nvidia-smi` command. Run again.
6. This concludes our demo for using an interactive session.

## Batch Jobs with SLURM:

We can use the same scripts that we have to submit a job to the cluster. The benefit is that we can submit the job and go away, and come back later and get our results. This is done through `sbatch` scripts

1. An example `sbatch` script is present at `example1/submit_job.sh`
2. Type `sbatch example1/submit_job.sh`. It'll submit a job to the cluster, and create an output file when done. We can inspect the output.
3. We can also inspect this file using `vi example1/submit_job.sh`. Notice that we pass flags similar to what we did in the `srun` session.
4. This concludes our demo for using batch jobs with SLURM.

## Requesting a Specific GPU Type

So far we have asked for `--gres=gpu:1`, which means "any one GPU". To ask for a specific type, put the GPU name (as shown by `sinfo -p <partition_name> -o "%G"`) in the middle:

```bash
# In an sbatch script
#SBATCH --gres=gpu:t4:1

# Or in an interactive session
srun --partition=gpu --gres=gpu:a100:1 --pty bash
```

The GPU names on the `gpu` partition include `t4`, `v100-sxm2`, `v100-pcie`, `a100`, and `h200`.

**Which one should I pick?**

| GPU | Good for | Things to consider |
|---|---|---|
| `t4` | Learning, debugging, inference, small models | Least powerful and least GPU memory, but usually the shortest wait |
| `v100-sxm2` / `v100-pcie` | Most general training jobs | A good middle ground |
| `a100` | Large models, large batch sizes | Much faster with more memory; in high demand, so expect longer waits |
| `h200` | Very large models that need a lot of GPU memory | Most powerful, and usually the longest wait |

> 💡 **Rule of thumb:** If any GPU will do, ask for `--gres=gpu:1` and let Slurm choose. It starts your job on whichever GPU frees up first. Ask for a specific type only when your job needs it (e.g. it needs more GPU memory). Asking for an A100 for a 5-minute test can mean waiting hours for a job a T4 would have finished already.

Once your job is running, `nvidia-smi` shows which GPU you got and how much memory it has.

## Resource Limits: What Happens When You Hit Them?

Slurm enforces the limits from the [partition limits table](#partition-limits). Here is what each one looks like:

| What happened | What you will see | What to do |
|---|---|---|
| Job ran longer than `--time` | Job is killed; state is `TIMEOUT` | Request more time (up to the partition max), or save checkpoints so the job can resume |
| Job used more RAM than `--mem` | Job is killed; state is `OUT_OF_MEMORY` | Request more `--mem` |
| Too many jobs submitted | `sbatch` refuses the job, or it waits in `squeue` with reason `QOSMaxSubmitJobPerUserLimit` / `job violates accounting/QOS policy` | Wait for some of your jobs to finish |
| Too many jobs running | Extra jobs wait (`PENDING`) with a reason like `QOSMaxJobsPerUserLimit` | Nothing to fix; they start once earlier jobs finish |
| Asked for more than the partition allows (e.g. `--time=12:00:00` on `gpu`) | Job is rejected or never starts | Lower the request, or pick a different partition |

To find out why a job ended or why it is still waiting:

```bash
# Why is my job still pending? (see the REASON column)
squeue -u $USER

# How did my finished job end?
sacct -j <jobid> --format=JobID,JobName,Partition,State,Elapsed,Timelimit,ExitCode
```

> **Note:** You may see `job violates accounting/QOS policy` even when you are under your own limit. This happens when your whole Slurm **account** (your group) hits the overall limit of 10,000 jobs. It goes away once some of those jobs start or finish.


## Monitoring Your Jobs

- To see how many resources are used by your current job, you can put `nvidia-smi` commands in your script.
- Use `gpu-logs <jobid>` to see GPU usage statistics for your completed jobs.

```bash
gpu-logs 2374457
                       GPU Info                       
                        GPU #2                        
            Time|    Memory|   GPU Utility|  Activity
=====================================================
2025-10-20T12:52|       0 %|           0 %|  inactive
2025-10-20T12:53|       0 %|           0 %|  inactive
2025-10-20T12:54|       0 %|           0 %|  inactive
2025-10-20T12:55|       0 %|           0 %|  inactive
2025-10-20T12:56|       0 %|           0 %|  inactive
2025-10-20T12:57|   22.56 %|           0 %|  inactive
2025-10-20T12:58|   22.56 %|           0 %|  inactive
2025-10-20T12:59|   22.56 %|           0 %|  inactive
2025-10-20T13:00|   22.56 %|           0 %|  inactive
2025-10-20T13:01|   22.56 %|           0 %|  inactive
2025-10-20T13:02|   77.98 %|           0 %|  inactive
2025-10-20T13:03|   77.98 %|           0 %|  inactive
2025-10-20T13:04|   77.98 %|           0 %|  inactive
2025-10-20T13:05|   77.98 %|           0 %|  inactive
2025-10-20T13:06|   77.98 %|           0 %|  inactive
2025-10-20T13:07|    81.1 %|          56 %|  inactive
2025-10-20T13:09|    81.1 %|          56 %|  inactive
2025-10-20T13:11|    81.1 %|          56 %|  inactive
```



## Using GPUs Efficiently

GPUs can be very powerful, but only if used correctly. Let's look at some examples of inefficient vs. efficient GPU code:

### Example 1: Batch Processing

**Naive Approach (Inefficient):** Processing data one sample at a time

```python
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Linear(2048, 2048).to(device)

data_size = 50000

start = time.time()
# Sending 50,000 separate requests to GPU - lots of overhead!
for _ in range(data_size):
    data = torch.randn(1, 2048).to(device)
    _ = model(data)

torch.cuda.synchronize()
print(f"Naive Time: {time.time() - start:.4f}s")
```

**Optimized Approach (Efficient):** Batching all data together

```python
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Linear(2048, 2048).to(device)

data_size = 50000

start = time.time()
# Send all data at once - much more efficient!
data = torch.randn(data_size, 2048).to(device)
_ = model(data)

torch.cuda.synchronize()
print(f"Optimized Time: {time.time() - start:.4f}s")
```

The optimized version is significantly faster because it reduces GPU overhead and improves parallelization.

### Example 2: Data Loading

**Naive Approach (Inefficient):** Single-threaded data loading

```python
import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# num_workers=0 means CPU loads data one-by-one (bottleneck!)
loader = DataLoader(dataset, batch_size=256, num_workers=0)

for images, labels in loader:
    images = images.to("cuda")
    # GPU sits idle waiting for data...
```

**Optimized Approach (Efficient):** Multi-threaded data loading with pinned memory

```python
import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# num_workers=2 uses multiple CPU cores to prepare data in parallel
# pin_memory=True speeds up CPU-to-GPU transfers
loader = DataLoader(dataset, batch_size=256, num_workers=2, pin_memory=True)

for images, labels in loader:
    # non_blocking=True allows GPU to process while CPU loads next batch
    images = images.to("cuda", non_blocking=True)
    # GPU stays busy!
```

**Key Takeaways:**
- Always batch your data instead of processing one sample at a time
- Use `num_workers` in DataLoader to parallelize data preprocessing
- Minimize idle time on the GPU - keep it working!

## Troubleshooting Common Errors

### `CUDA out of memory`

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB ...
```

**Why:** Your model, data, and intermediate results don't fit in the GPU's memory. GPU memory is separate from the `--mem` RAM you request.

**What to do:**
- Lower your batch size. This is the most common fix.
- Run `nvidia-smi` while the job runs to see how much memory the GPU has and how much is in use.
- During evaluation, wrap your code in `with torch.no_grad():` so PyTorch doesn't keep gradients you won't use.
- If it still doesn't fit, request a GPU with more memory (see [Requesting a Specific GPU Type](#requesting-a-specific-gpu-type)).

### No GPU found / `torch.cuda.is_available()` returns `False`

**Why:** Usually no GPU was allocated to your job. For example:
- You are running on a **login node** instead of inside an `srun` session or `sbatch` job.
- You left out `--gres=gpu:1`. Choosing a GPU partition alone does not give you a GPU.

**What to do:** Check that Slurm actually gave you a GPU:
```bash
echo $CUDA_VISIBLE_DEVICES   # should print a number, e.g. 0
nvidia-smi                   # should list your GPU
```
If both come up empty, request a new session with `--gres=gpu:1`.

## Best Practices for Using GPUs on Explorer
- **Choose the Right Partition:** Select the appropriate GPU partition based on your job requirements.
  - If you have a short job, use `gpu-short` for faster scheduling.
  - Use `sbatch` for longer running gpu jobs.
- **Optimize Resource Requests:** Request only the resources you need.
    - To see how many cpus you need, you can use `seff <jobid>` on a previously completed job.
- **Use Efficient Libraries:** Utilize GPU-optimized libraries (e.g., CUDA, cuDNN) for better performance.
- **Profile Your Code:** Use profiling tools to identify bottlenecks and optimize your GPU code. 

## How to get help

Email the Research Computing team at [rchelp@northeastern.edu](mailto:rchelp@northeastern.edu).

Come to [office hours](https://rc.northeastern.edu/getting-help/) hosted on Zoom.

Or [book a consultation](https://rc.northeastern.edu/getting-help/) with an RC team member.

Review our [Documentation](https://rc-docs.northeastern.edu/en/latest/index.html).

Thank you!

---

*For questions or support, contact the Research Computing team at [rchelp@northeastern.edu](mailto:rchelp@northeastern.edu)*
