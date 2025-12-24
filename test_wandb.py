import wandb

wandb.init(
    project="xai-project",
    entity="junsche-university-of-bamberg",
    name="new-account-test"
)

for epoch in range(5):
    wandb.log({"epoch": epoch, "loss": 1.0 / (epoch + 1)})

wandb.finish()