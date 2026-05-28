# HandDigit AI — MNIST Handwritten Digit Classifier

An interactive handwritten digit classifier built with TensorFlow and Streamlit. Train a neural network on the MNIST dataset, upload your own digit images for prediction, and teach the model from your corrections — all through a clean web interface.

---

## Repo name suggestion

**`handdigit-ai`** or **`mnist-interactive-classifier`**

## Description (for GitHub)

> Interactive MNIST digit classifier with online learning. Train a neural network, predict handwritten digits from uploaded images, and improve the model in real time by correcting its mistakes — changes are saved permanently to disk.

---

## Project structure

```
├── main.ipynb           # Jupyter notebook — exploration & training walkthrough
├── app.py               # Streamlit web app
├── mnist_model.keras    # Saved model (created after first train)
├── corrections_log.json # Persistent feedback history (created after first correction)
└── requirements.txt     # (see Dependencies below)
```

---

## Dependencies

| Package | Version used |
|---|---|
| tensorflow | 2.21.0 |
| streamlit | 1.57.0 |
| pillow | 12.2.0 |
| scikit-learn | 1.7.2 |
| seaborn | 0.13.2 |
| matplotlib | 3.10.9 |
| numpy | 2.2.6 |

Install all at once from the requirements file:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow streamlit pillow scikit-learn seaborn matplotlib numpy
```

Or with the virtual environment already in this repo:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

## Jupyter Notebook — `main.ipynb`

The notebook is a step-by-step walkthrough of the full ML pipeline. Run cells top to bottom.

### Cell 1 — Imports
Loads TensorFlow, NumPy, Matplotlib, Seaborn, and scikit-learn.

### Cell 2 — Train the model
- Downloads the MNIST dataset (60 000 training images, 10 000 test images)
- Normalises pixel values to 0–1
- Builds a 3-layer neural network:
  - `Flatten` — converts each 28×28 image to a flat 784-value vector
  - `Dense(128, relu)` — hidden layer with 128 neurons
  - `Dropout(0.2)` — randomly disables 20 % of neurons each step to prevent overfitting
  - `Dense(10, softmax)` — output layer, one neuron per digit (0–9), outputs a probability per class
- Trains for 5 epochs with 20 % validation split
- Evaluates on the test set (~97 % accuracy)

### Cell 3 — EDA (Exploratory Data Analysis)
- Bar chart showing how many samples exist per digit class
- Grid of 10 sample images with their labels

### Cell 4 — Confusion matrix & classification report
- Runs predictions on all 10 000 test images
- Plots a heatmap of the confusion matrix — rows are true labels, columns are predicted labels
- Prints per-class precision, recall, and F1-score

### Cell 5 — Training curves
- Side-by-side line plots of accuracy and loss for both the training and validation sets across all 5 epochs

### How to run

1. Open the notebook in Jupyter Lab, Jupyter Notebook, or VS Code
2. Select your Python environment (the one with the packages above)
3. Run all cells: **Kernel → Restart & Run All**

---

## Streamlit App — `app.py`

A full web interface for the model with four pages accessible from the left sidebar.

### How to start

```bash
# activate the environment first, then:
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

### Page 1 — Predict

Upload a photo of a handwritten digit and the model will predict what number it is.

**Steps:**
1. Click **Browse files** and upload a PNG or JPG image
2. The app shows your original image, the 28×28 version the model sees, and the predicted digit with confidence
3. A bar chart shows the probability for each digit 0–9

**Giving feedback (teaches the model):**
- Click **✓ Yes, correct!** if the prediction was right
- Select the correct digit from the dropdown and click **✗ Correct & teach model** if it was wrong

Every time you give feedback the model weights are updated immediately and **saved permanently to disk** (`mnist_model.keras`). All corrections are also logged to `corrections_log.json` with a timestamp. The model remembers everything even after restarting the app.

**Tips for best results:**
- Write the digit large, centred, with a dark pen on white paper
- Take a well-lit, straight-on photo with no shadows
- Crop tightly around the digit before uploading

---

### Page 2 — Train Model

Train a fresh model from scratch on the full MNIST dataset.

1. Use the **Epochs** slider to choose how long to train (5 is a good default, more epochs = slower but potentially higher accuracy)
2. Click **Train / Retrain**
3. Watch the live progress bar update each epoch with accuracy and loss values
4. Accuracy and loss curves are plotted after training completes

> **Note:** Retraining creates a brand-new model and overwrites any corrections you have taught it. The correction history in `corrections_log.json` is kept, but the weight updates are lost.

---

### Page 3 — EDA

Explore the MNIST dataset visually without writing any code.

- **Class distribution** — bar charts for training and test sets showing sample counts per digit
- **Sample images** — a 4×5 grid of random training images; use the dropdown to filter by a specific digit
- **Average pixel intensity** — a heatmap showing the average appearance of each digit class, useful for seeing what the model has "learned to look for"

---

### Page 4 — Evaluate

Measure how well the current model performs on the 10 000 held-out test images.

- **Test accuracy** metric
- **Confusion matrix** heatmap — spot which digits the model confuses most often
- **Classification report** table — per-digit precision, recall, and F1-score
- **Most confused pairs** — the top 5 digit combinations the model gets wrong most frequently

---

## How the online learning works

When you correct the model on the Predict page, the app calls `model.train_on_batch()` several times on your single image with the correct label. This is called **online / incremental learning** — the model updates its weights on the fly without retraining on the full dataset.

A confirmed correct prediction runs **3 gradient steps** (reinforcement).  
A correction runs **6 gradient steps** (stronger update to override the wrong answer).

The updated model is saved to `mnist_model.keras` immediately after each update. On the next app start, this file is loaded automatically — corrections are never lost.

> **Keep in mind:** fine-tuning on single images is powerful for quick adaptation but can cause *catastrophic forgetting* if you feed the model many corrections from only one or two digit classes. For best results, spread corrections across different digit types and retrain from scratch periodically to reset a clean baseline.
