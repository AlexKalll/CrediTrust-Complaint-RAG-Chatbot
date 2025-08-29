#!/usr/bin/env python3
"""
Startup script for the CreditTrust RAG Chatbot.
Allows users to choose between Streamlit and Gradio interfaces.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("🏦 CreditTrust RAG Chatbot")
    print("=" * 40)
    print("Choose your interface:")
    print("1. Streamlit (Recommended - Better UI)")
    print("2. Gradio (Alternative)")
    print("3. Run setup test")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            print("\n🚀 Starting Streamlit app...")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "dashboards/streamlit_app.py",
                    ],
                    check=True,
                )
            except KeyboardInterrupt:
                print("\n👋 App stopped by user")
            except Exception as e:
                print(f"❌ Error starting Streamlit: {e}")
            break

        elif choice == "2":
            print("\n🚀 Starting Gradio app...")
            try:
                subprocess.run([sys.executable, "dashboards/app.py"], check=True)
            except KeyboardInterrupt:
                print("\n👋 App stopped by user")
            except Exception as e:
                print(f"❌ Error starting Gradio: {e}")
            break

        elif choice == "3":
            print("\n🔍 Running setup test...")
            try:
                subprocess.run([sys.executable, "test_setup.py"], check=True)
            except Exception as e:
                print(f"❌ Error running test: {e}")
            break

        elif choice == "4":
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
