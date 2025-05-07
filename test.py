share_link = 'https://drive.google.com/file/d/13-_kf4MYqIdNshJCEr9ishnnV23SbEvk/view?usp=drive_link'

import gdown

file_id = "13-_kf4MYqIdNshJCEr9ishnnV23SbEvk"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "file.csv", quiet=False)
