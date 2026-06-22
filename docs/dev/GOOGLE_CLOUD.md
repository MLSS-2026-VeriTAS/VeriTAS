# RUNNING VM ON GOOGLE CLOUD

0. To run the Google Cloud VM, first you will need to be invited to the Google Cloud project. The owner will need to edit IAM Permissions to give you necessary permissions.

1. Install gcloud on your local machine. See [installation instructions](https://docs.cloud.google.com/sdk/docs/install-sdk).

2. In terminal run `gcloud compute ssh --zone "us-central1-b" "instance-20260622-053006" --project "veritas-500122"`

3. Install conda. See [installation instructions](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install).


———

tmux a -t 0

conda activate product-recommendation

export GCP_PROJECT_ID="veritas-500122"
export TASK_NAME=product-recommendation
export MODEL=gemini-1.5-flash-002
export GPU_ID=0


 bash scripts/init_env.sh ${TASK_NAME} ${MODEL} ${GPU_ID} "TEST_MODEL"

bash launch.sh ${TASK_NAME} ${MODEL} ${GPU_ID}