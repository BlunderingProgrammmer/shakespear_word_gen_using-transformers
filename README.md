# Trigram Language Model Text Generator

## Description
This project implements a trigram language model using PyTorch and provides a Streamlit web application to generate text based on user input prompts. The model is trained on a custom text dataset and can generate coherent text sequences by predicting the next character given the previous two characters.

## Installation

1. Clone the repository.
2. Install the required Python packages:
```bash
pip install torch streamlit
```

## Usage

1. Ensure you have the trained model file `trigram_model.pth` in the project directory. If not available, you can train the model by running `test.py`.
2. Run the Streamlit app:
```bash
streamlit run streamlit_dashboard.py
```
3. Open the URL provided by Streamlit in your browser.
4. Enter a prompt in the input box and click "Generate" to see the generated text.

## Model Details

- The model is a trigram language model implemented with a transformer architecture.
- It uses character-level embeddings and positional embeddings.
- The model is trained to predict the next character based on the previous two characters.
- Training parameters such as batch size, learning rate, and number of iterations are defined in `test.py`.

## File Descriptions

- `test.py`: Contains the model implementation, training loop, and utilities for encoding/decoding text.
- `streamlit_dashboard.py`: Streamlit app that loads the trained model and provides a user interface for text generation.
- `trigram_model.pth`: The saved trained model weights (not included in the repository).

## License

This project is licensed under the MIT License.
