# Cloud Options
We list various cloud options and how to set them up. The cheapest option is RunPod.

## AWS (Amazon Web Services)

### How to set up

**Step 1: Launch a GPU Instance**

1. **Log into AWS**: Go to the [AWS Management Console](https://aws.amazon.com/console/). Make an account or log in if you have one.
2. **Create a new EC2 instance**:
    * Go to **EC2** > **Launch Instance**.
    * Choose a name for the instance.
    * Select an **AMI (Amazon Machine Image)** that is Unix based, supports GPU, and has CUDA and PyTorch installed. For example,
        * `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)`.
    * **Choose an Instance Type**:
       * Select an instance type with a **GPU** that has at least 16GB VRAM and 32GB RAM, e.g. `g4dn.2xlarge` which has an NVIDIA T4
    * Choose a key pair
        * If you do not have a key pair, click **Create new key pair** and download the **.pem key**
3. **Configure the Instance**:
    * Set up **security groups** to allow SSH (port 22) and a specific port for protocol communication (port 49200, though you can change and specify this)
    * We recommend minimum of 80GB of storage
1. **Review and Launch** the instance.

**Step 2: Edit Security Group**
1. Follow the AWS section in the [network guide](network.md) to configuring port 49200 to be accessible for external connections.

**Step 3: Connect to the Instance**

1. **SSH into your instance**:
    * Find your public ip and connect via SSH:
     ```bash
     ssh -i your-key.pem ubuntu@your-ec2-public-ip
     ```

## GCP (Google Cloud Platform)

The cheapest option is a `NVIDIA T4` (16GB VRAM) and `n1-standard-8` (8 vCPU, 4 core, 30 GB memory) for $0.51/hour.

### How to set up

**Step 1: Create a GPU-enabled VM**

1. **Log into Google Cloud** Go to the [Google Cloud Console](https://console.cloud.google.com/). Make an account or log in if you have one.
2. **Create a new VM instance**:

    * Go to **Compute Engine** > **VM instances** > **Create Instance**.
    * Choose a name and region for the instance.
    * Change from General Purpose to **GPUs** and select a **GPU** and **Machine Type**
        * E.g. Choose 1 `NVIDIA T4` and `n1-standard-8` (8 vCPU, 4 core, 30 GB memory)
    * In the **OS and storage** tab change the image to one that is Unix based, supports GPU, and has CUDA and PyTorch installed
        * E.g. OS `Deep Learning on Linux` and image `Deep Learning VM for PyTorch 2.4 with CUDA 12.4 M129`
    * In the **Security** tab click **Manage Access**. Under **Add manually generated SSH keys** click **Add item**, enter your SSH public key, and click **Save**.
    * Click **Create**

**Step 2: Edit Firewall settings**
1. Follow the GCP section in the [network guide](network.md) to configuring port 49200 to be accessible for external connections.

**Step 2: Connect to the Instance**

1. **Set up SSH Keys**
    * Go into your instance and click **Edit**
    * Under **SSH Keys** click **Add item**, enter your SSH public key, and click **Save**.
2. **SSH into your VM**:
    * Find your external ip and username (this is linked with your SSH key) under the instance details and connect via SSH:
     ```bash
     ssh ubuntu@your-external-ip
     ```


## RunPod

The cheapest option is a RTX 2000 Ada: 16GB VRAM, 31Gb RAM, 6 vCPUs for $0.23/hour.

RunPod launches your workspace within a docker container, so it is difficult to launch docker within the docker container.
We recommend using conda instead. See the [installing guide](installing.md) for how to install conda.

RunPod also assigns random external port mappings, so we need to find and specify that external port. See the RunPod section in the [network guide](network.md)

Finally, if you need to install anything else with RunPod, note that most standard packages are not installed, so run `apt update` first.

### How to set up
**Step 1: Launch a GPU Pod**

1. **Log into RunPod**: Go to the [RunPod Console](https://www.runpod.io/console/home). Make an account or log in if you have one.
2. **Set SSH Keys**:
    * Go to **Settings** and under **SSH Public Keys** add your public SSH key. If you have not made a SSH key yet, follow [this guide from RunPod](https://docs.runpod.io/pods/configuration/use-ssh).
3. **Create a new Pod**:
    * Go to **Pods** to see available pods, and choose a Pod
        * E.g. RTX 2000 Ada: 16GB VRAM, 31Gb RAM, 6 vCPUs for $0.23/hour.
    * Choose a Pod name and **Pod Template** 
        * Want one with CUDA and PyTorch installed, the default `RunPod Pytorch 2.1` works.
    * Ensure that SSH Terminal Access is enabled
    * Click **Deploy On-Demand**

**Step 2: Edit the Pod**
1. Follow the RunPod section in the [network guide](network.md) to edit the Pod to expose a TCP port.

**Step 3: Connect to the Pod**

1. **SSH into your Pod**:
   * Go to **Connect** and in the **SSH** tab look at the ssh command under **SSH over exposed TCP**.


## Tensordock

Tensordock offers low-cost consumer GPUs as low as an RTX A4000 for $0.105/hr.

The distributed compute option in Tensordock also assigns random external port mappings, so we need to find and specify that external port. See the Tensordock section in the [network guide](network.md)

### How to set up
1. **Log into Tensordock**: Go to the [Tensordock Deploy Dashboard](https://dashboard.tensordock.com/deploy). Make an account or log in if you have one.
2. **Set SSH Keys**:
    * Go to **Secrets** and click **Add Secret** to add your public SSH key. If you have not made a SSH key yet, follow [this guide for Windows](https://learn.microsoft.com/en-us/viva/glint/setup/sftp-ssh-key-gen) and [this guide for Linux](https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server). 
    * Choose a **Name** for your SSH Key, choose **Type** as `SSH Key` and enter your public key value under **Value**. The public key value will look something like this `ssh-rsa ...`
3. **Deploy a GPU**
    * Go to **Deploy GPU** to see available GPUs, and choose a GPU
        * E.g. RTX 4000: 16GB VRAM for $0.105/hour.
    * Choose a Instance Name, configure the resource with CPU Cores, RAM and Storage options and choose a location and select the OS. We recommend `Ubuntu 24.04 LTS`.
    * Click **Deploy Instance**
4. **Connect to your instance**
    * Click on **My Servers** and you should see the newly provisioned GPU instance. You can click the instance to get details about the instance
    * Instructions for connecting to the instance using SSH can be found under the **Access** section

#### CUDA \& Docker setup
You may need to setup your Tensordock instances with NVIDIA toolkit and Docker (if using Docker).

To install NVIDIA toolkit, run the following commands in your instance CLI:
```
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

To install Docker, run the following commands in your instance CLI:
```
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Lambda Labs
Lambda does not support 16GB GPUs, the cheapest options is a RTX 6000 (24 GB VRAM) with 14 vCPUs, 46 GiB RAM, 0.5 TiB SSD for $0.50 / hr.

### How to set up
**Step 1: Launch a GPU Instance**

1. **Log into Lambda**: Go to the [Lambda instances](https://cloud.lambda.ai/instances). Make an account or log in if you have one.
2. **Set SSH Keys**:
    * Go to **SSH Keys** and add your public SSH key.
3. **Create a new Instance**:
    * Go to **Instances** and select **Launch an Instance** to see available instances, and choose an instance
        * E.g. 1x RTX 6000 (24 GB), for $0.50/hour.
    * Choose a **Region** and **FileSystem**. If you don't have a filesystem, select **Create a filesystem**
    * Click **Launch**

**Step 2: Edit the Firewall**
1. Follow the Lambda Labs section in the [network guide](network.md) to edit the firewall to expose a TCP port.

**Step 3: Connect to the Instance**

1. **SSH into your Instance**:
    * Once the instance has booted, look at the SSH command under **SSH Login**.

