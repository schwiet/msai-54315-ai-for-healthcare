# This is a scratch file that I can use to look at notes
import pandas as pd

chunksize = 10000
chunks = []

# Iterate through the file in chunks
for chunk in pd.read_csv("./mimic-iii/NOTEEVENTS.csv.gz", chunksize=chunksize):
    # FILTER: Keep only discharge summaries
    filtered_chunk = chunk[chunk['CATEGORY'] == 'Discharge summary']
    
    # STORE: Add the filtered piece to our list
    chunks.append(filtered_chunk)

# Glue all the pieces together
discharge_summaries = pd.concat(chunks)
discharge_summaries.info()

def compare_patient_notes(id_1, id_2):
    # filter the dataframe to find rows matching the Subject ID
    # use .values[0] to grab the text from the first matching row
    try:
        note_1 = discharge_summaries[discharge_summaries['SUBJECT_ID'] == id_1]['TEXT'].values[0]
    except IndexError:
        note_1 = "Note not found"

    try:
        note_2 = discharge_summaries[discharge_summaries['SUBJECT_ID'] == id_2]['TEXT'].values[0]
    except IndexError:
        note_2 = "Note not found"
    
    print(f"--- PATIENT {id_1} (Query) ---")
    print(note_1[:500] + "...") 
    print("\n" + "="*50 + "\n")
    
    print(f"--- PATIENT {id_2} (Match) ---")
    print(note_2[:500] + "...")
    print("\n")

# look at notes for two patients
compare_patient_notes(19, 19835)