# Typical failure cases in Discovery and how to address them (plus some best practices)
<!-- TOC -->

- [Typical failure cases in Discovery and how to address them plus some best practices](#typical-failure-cases-in-discovery-and-how-to-address-them-plus-some-best-practices)
    - [How do I submit a ticket?](#how-do-i-submit-a-ticket)
    - [How to make it easier for yourself to find out about issues](#how-to-make-it-easier-for-yourself-to-find-out-about-issues)
    - [How to check for available resources](#how-to-check-for-available-resources)
    - [Jobs not getting allocated despite resources being available case 1](#jobs-not-getting-allocated-despite-resources-being-available-case-1)
    - [Jobs not getting allocated despite resources being available case 2](#jobs-not-getting-allocated-despite-resources-being-available-case-2)
    - [Jobs not getting allocated despite resources being available case 3](#jobs-not-getting-allocated-despite-resources-being-available-case-3)
    - [Jobs dying suddenly](#jobs-dying-suddenly)
        - [Case 1: no storage](#case-1-no-storage)
        - [Case 2: your script has an error](#case-2-your-script-has-an-error)
        - [Case 2.1: your script dies... but only sometimes](#case-21-your-script-dies-but-only-sometimes)
        - [Case 3: no GPUs assigned](#case-3-no-gpus-assigned)
        - [Case 4: the whole node fails](#case-4-the-whole-node-fails)
    - [Bonus: some best practices](#bonus-some-best-practices)
        - [Use config files](#use-config-files)
        - [Making sure your job is using the GPUs you requested](#making-sure-your-job-is-using-the-gpus-you-requested)
        - [Requesting a job without GPUs](#requesting-a-job-without-gpus)
        - [Setting the number of CPUs per GPU](#setting-the-number-of-cpus-per-gpu)
        - [Setting time limits for interactive sessions](#setting-time-limits-for-interactive-sessions)
        - [Limiting your use of resources while still submitting everything you need](#limiting-your-use-of-resources-while-still-submitting-everything-you-need)

<!-- /TOC -->

Sometimes Discovery won't work as expected, if you notice or suspect it, it's best to report it either to maintenance or to the rest of the lab so that people are aware of it. Here I will detail some easy to detect failure cases and others that aren't quite as obvious.

Note: this is not intended to be a tutorial, for such purposes please read the docs: <https://rc-docs.northeastern.edu/en/latest/index.html> or check the course available on Canvas: <https://rc-docs.northeastern.edu/en/latest/tutorialsandtraining/canvasandgithub.html#canvasandgithub>

## How do I submit a ticket?

This is the simplest part: just send an email to **rchelp (at) (neu's domain) (dot) edu** with the relevant information and this will open a ticket in the system. If the issue seems to be big, you can CC Prof. Jiang to keep him in the loop.

## How to make it easier for yourself to find out about issues

Always define your jobs with the following flags:

```slurm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<YOUREMAIL>@northeastern.edu
#SBATCH --output=./logs/exec.%j.%x_inf.out
#SBATCH --error=./logs/exec.%j.%x_inf.out
```

(note: these flags can also be added to interactive sessions)

The mail field will send you an email upon jobs starting, being cancelled, completing or failing. This is very useful to keep track of your jobs and can sometimes allow you to identify why they failed without having to look into your log files. This can fill your email with notifications but you can define  rules so they get added to some folder or in time they will get automatically added to the 'Other' section if you're using Outlook.

## How to check for available resources

Run `sinfo -p jiang --Format=nodes,cpus,nodelist,gres,nodeaiot,GresUsed`, the last column will tell you how many GPUs are under utilization.

## Jobs not getting allocated despite resources being available (case 1)

This usually happens when someone requests too much RAM or CPUs. People often don't do this on purpose, a typical scenario is someone had a command or script to request 4 GPUs with 12 CPUs per GPU and changed only the amount of GPUs so either the job requires more CPUs than are available and does not get allocated or the job will get allocated and block other jobs from using GPUs because not enough CPUs are available. If you suspect this is happening in a specific node, run the following:

```bash
squeue -p jiang --format="%.18i %.9P %.60j %.8u %.2t %.10M %.6D %R"
```

You should see a list of jobs and corresponding nodes they're running on, as well as their usernames. You can then run:

```bash
scontrol show jobid <JOB_ID_HERE>
```

This will show you the job, look for the `ReqTRES` field to see the resources requested by the job.

Ideally, all jobs in the `jiang` partition should use at most 12 CPUs per 1 GPU. If you see that the number of CPUs is greater than this, reach out to the person whose job is using these CPUs and kindly ask them to resubmit it using less CPUs if possible.

On the other hand, if your job definitely requires that many CPUs, consider either using the 'short' partition in case its GPU usage is negligible (for example, if you're doing data pre-processing) or let the rest know in the lab's `gpus` Slack channel that you will be using that many (for example if your workload requires more than 12 workers per GPU.)

PS: You can check for resource availability using `sinfo -p jiang --Format=nodes,cpus,nodelist,gres,nodeaiot,GresUsed`

## Jobs not getting allocated despite resources being available (case 2)

If you run `squeue` you will see jobs waiting for resources are in the top. These jobs are ordered such that the jobs on top have more priority (due to SLURM's scheduling algorithm.) Two typical states are `priority` and `resources`. The latter means a job is going to be next allocated resources when these are available.

Sometimes, there will be jobs requesting 8 GPUs. These jobs will have lower priority in the queue due to their high resource usage but at some point they will go into `resources` state (this may apply even in `priority`, I'm not sure) at which point no further jobs will be allowed to get the remaining resources even if they're available. Typically this will look like this:

- 1 job using 4 GPUs in node dXXXX.
- 1 job requesting 8 GPUs in `resources` state.
- 2+ jobs waiting without being able to use the 4 remaining GPUs.

This may be annoying but sometimes people really need to use 8 GPUs so unless there is some huge deadlock like the currently running job being scheduled to use the 4 GPUs for 4 days, it's better to just wait (also, most of the time the scheduler will handle these cases on its own anyway)

For more on the scheduler, see: <https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html>

## Jobs not getting allocated despite resources being available (case 3)

From the `sinfo` command, look at the `NODES(A/I/O/T)` column. If a node is in state `O` that can mean it is down. Report it to Discovery ASAP.

## Jobs dying suddenly

### Case 1: no storage

If there is no storage then your program won't be able to write anything and things will crash. This error case is difficult to track because error messages can be a bit cryptic, so it's good to check for this every time to avoid wasting time. Either your home (`~`) directory is full or the lab's storage is full, you can check this by running:

```bash
du -h ~ --max-depth=1 # for your home directory, should be under 50 GB
df -h /work/vig # for the lab's storage, look for the Avail column
```

In the former case, just delete some files (`conda clean --all` always helps), in the latter, see if you can delete your own files or report this in `gpus` so that everybody can delete old stuff. If your `conda` environments

### Case 2: your script has an error

This is the most typical case, you will have to debug your code. Maybe use an interactive session for this when that happens so you don't have to wait between jobs. If your script runs well in an interactive session but not in a SLURM job then potential cases include:

- your Python environment is not loading properly with `conda` (try replacing `conda` by `source`) and remember to load  the anaconda module inside your SLURM script with `module load anaconda3/2022.05`
- your code relies on the CUDA install from Discovery and you didn't load it in the SLURM script, do it with `module load cuda/XX.X` where XX.X is the version you're using
- you forgot to set `PYTHONPATH` to point to your project files: `export PYTHONPATH=${WORKSPACE_PATH}`
- you're not exposing GPUs correctly with `CUDA_VISIBLE_DEVICES`
- you modified your code or your config files while the job was queued. Be aware that submitting a job only saves the state of the submit file, so if you play with the config files or the code before the job runs then the job will use the code/config files as they are when you the job gets the resources
  - prevent this issue by using Git branches in different folders or carefully putting your code for new features inside `if` statements that can only be accessed with newly defined configuration fields that default to `False`; also, always use config files that can have specific fields overriden through command line arguments
- your job uses multiple GPUs and is trying to use the default port for several jobs. This is library specific:
  - for vanilla Torch see: <https://pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker>
  - for `detectron2` you can use the `--dist-url` flag and set it to a random number such as `ID_PORT=$(($RANDOM+20001))` ->  `--dist-url "tcp://127.0.0.1:$ID_PORT"\`

### Case 2.1: your script dies... but only sometimes

I've noticed that my conda environments don't launch properly (imports fail, etc) when I run them from some nodes, particularly those from the public partitions. To avoid this, try to run your jobs either from `login` or from one of our own nodes.

### Case 3: no GPUs assigned

This can happen when some GPUs in the node get disconnected or crash, so sometimes your jobs won't get any GPUs. Look for errors like `RuntimeError: CUDA error: invalid device ordinal`. This kind of error should be reported to Discovery support ASAP with a ticket including the name of the node (or at least the jobid) has the issue. It *won't* get solved by itself and usually only requires a small reboot, so you can coordinate with the rest of the lab to allow support to do this if required.

### Case 4: the whole node fails

This can happen either under heavy load or other conditions. If this happens once then you can re-queue your jobs, but if it happens two days in a row you should report it to Discovery. Keep an eye out for error messages like:

```bash
Slurm Job_id=<JOB_ID> Name=JOB_NAME Failed, Run time d-hh:mm:ss, NODE_FAIL, ExitCode 0 # for email notifications

slurmstepd: error: *** STEP 41246761.0 ON <NODE_ID> CANCELLED AT YYYY-MM-DDThh:mm:ss DUE TO NODE FAILURE, SEE SLURMCTLD LOG FOR DETAILS *** # for log files
```

In these cases, please make a note of either the JOB_ID or the JOB_NAME and send a ticket to Discovery support so they can investigate the issue.

## Bonus: some best practices

### Use config files

This will save you precious time, set up your jobs with a unified default configuration (like `config.py` or `base.yaml`) that can be overriden with a config file (like a `experiment.yaml` file) which itself should allow you to override config fields from command line arguments.

Good libraries for this include:

- `pyyaml+argparse`: basic setup to read yaml files, not super flexible but can get the job done
- `yacs`: basic, somewhat flexible, well-documented
- `hydra`: flexible, well-documented
- `gin-config`: even more flexible, poor documentation

### Making sure your job is using the GPUs you requested

After a job gets resources you can SSH into the node with `ssh <USERNAME>@<NODENAME>` from the Login partition and run `nvidia-smi` or `watch -n 1 nvidia-smi`. Make sure that your job is using the GPUs properly! If your job uses 20 GB memory per GPU then you shouldn't be using A6000 GPUs, switch to A5000s (or just use less GPUs).

Note: SSH-ing like this this is only the case if you're running a single job in the specified node. Otherwise it's not guaranteed that you will get SSH'd into the node you want. An alternative is to set up your Python code to periodically run the `nvidia-smi` process and log the outputs. See: <https://stackoverflow.com/questions/706989/how-to-call-an-external-program-in-python-and-retrieve-the-output-and-return-cod>

### Requesting a job without GPUs

Sometimes you may want to request an interactive node to code or to run some pre-processing. While you can do this on the `short` partition, an alternative way to do it on our own partition is as follows:

`srun -p jiang --nodes=1 --cpus-per-task=2 --gres=gpu:a5000:0 --mem=16G --time=08:00:00  --pty ~/bin/bash`

Do remember that CPUs are still needed for other people so try to keep your jobs to at most 2 CPUs. Also, if a node is under heavy use your interactive session might run very slowly, so again, `short` is usually the best option here.

### Setting the number of CPUs per GPU

Use the flag `--cpus-per-gpu=12` to automatically scale the CPUs your job uses to the appropriate amount. Remember, each node has 96 cores so 12 should be the maximum here.

### Setting time limits for interactive sessions

Unless you're on a deadline and have agreed to get a specified number of nodes, keep your interactive jobs to a maximum of 8 hours. If your code finishes running in an interactive session then the job will remain idle so compute time will be wasted. You can limit time like this:

`srun -p jiang --nodes=1 --cpus-per-task=12 --gres=gpu:a5000:1 --mem=64G --time=08:00:00  --pty ~/bin/bash`

### Limiting your use of resources while still submitting everything you need

If you need to queue many jobs at a time that have the same job_name but different parameters use the `--dependency=singleton` so that only a single job from a given family of jobs runs at a time. More complex behaviors can be achieved with SLURM arrays: <https://rc-docs.northeastern.edu/en/latest/slurmguide/slurmarray.html>

Another option to avoid saturing the cluster while running too many independent jobs is using the `#SBATCH --nice=X` flag. If you use X=0 your job will have normal priority, using values greater than 0 will give it decreased priority. This way your jobs will allow for more of other people's jobs to run before getting resources and you won't have to keep your eyes glued to the queue.

More on SLURM flags: <https://slurm.schedmd.com/sbatch.html>
