import questionary
from rich import print
from tui.dataset import load_dataset_flow
from tui.training import train_model_flow
from tui.inference import inference_flow


def main_menu():
    return questionary.select(
        "What would you like to do?",
        choices=["Load Dataset", "Train Model", "Run Inference", "Exit"],
    ).ask()


def run_app():
    while True:
        choice = main_menu()
        if choice == "Load Dataset":
            load_dataset_flow()
        elif choice == "Train Model":
            train_model_flow()
        elif choice == "Run Inference":
            inference_flow()
        elif choice == "Exit":
            print("Goodbye!")
            break

        input("\nPress Enter to return to main menu...")


if __name__ == "__main__":
    run_app()
