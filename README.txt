CAT VS DOG IMAGE CLASSIFICATION — HOW TO RUN

FILES
- Cat_Dog_Image_Classification.ipynb : complete Google Colab training notebook
- requirements.txt                   : libraries for the local website
- app.py                              : Streamlit image-classification website
- models/                             : put the trained model files here

PART A — TRAIN THE MODEL IN GOOGLE COLAB
1. Open https://colab.research.google.com/
2. Choose File > Upload notebook.
3. Upload Cat_Dog_Image_Classification.ipynb.
4. In Colab, choose Runtime > Change runtime type > T4 GPU (if available).
5. Run the notebook from the first cell using Runtime > Run all.
6. Upload your Cats and Dogs ZIP file to Google Drive (MyDrive).
7. In notebook Section 2, edit DRIVE_DATASET_PATH to the exact ZIP path.
   Example: /content/drive/MyDrive/Cats and Dogs.zip
   If the dataset is already extracted in Drive, set the path to that folder instead.
8. Run the Section 2 cell and approve the Google Drive connection.
   If using a ZIP, Colab extracts it into its temporary runtime storage.
9. Run the remaining notebook cells in order. Training can take a while.
8. At the final cell, download cat_dog_model_files.zip.
9. Extract that ZIP. It contains cat_dog_model.keras, model_info.json and model_comparison.csv.

PART B — PREPARE THE WEBSITE
1. Install Python 3.10 or 3.11 on your computer.
2. Extract this project folder.
3. Copy cat_dog_model.keras and model_info.json into the project's models folder.
   The final layout should be:
       project folder/
           app.py
           requirements.txt
           models/
               cat_dog_model.keras
               model_info.json
4. Open the project folder in VS Code.
5. Open Terminal > New Terminal.
6. Create a virtual environment:
       python -m venv .venv
7. Activate it on Windows:
       .venv\Scripts\activate
   On macOS/Linux:
       source .venv/bin/activate
8. Install the requirements:
       python -m pip install --upgrade pip
       pip install -r requirements.txt
9. Start the website:
       streamlit run app.py
10. Your browser should open the website. If not, open the Local URL shown in the terminal
    (usually http://localhost:8501).

TROUBLESHOOTING
- If Streamlit is not recognized, activate the virtual environment and run:
      python -m streamlit run app.py
- If the model file is missing, check that it is inside models/ and named exactly
  cat_dog_model.keras.
- Do not run app.py before the Colab training has produced the model file.
- If your dataset ZIP has a different structure, inspect its folders. The notebook searches
  recursively for folders named Cat/Cats and Dog/Dogs.
- The notebook includes traditional ML experiments that can take time. If a Colab session
  runs out of memory, reduce ML_MAX_PER_CLASS and HOG_MAX_PER_CLASS in the relevant cells.

NOTE
This is a student capstone implementation. Actual dataset counts, scores and model performance
are calculated by running the notebook; they are not pre-filled or guaranteed.
