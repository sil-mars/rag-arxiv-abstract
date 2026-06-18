import json
import pandas as pd

class Loader:

  def __init__(self, p):
    self.path = p

  def clean_data(self, text):
      text = str(text) # transforma todo en str
      text = text.replace("\n", " ") # quitar saltos de linea
      text = " ".join(text.split()) # normaliza todos los espacios, separando palabras y luego separandolas por espacio
      return text

  def preprocessing(self, item):
    return self.clean_data(item["title"]) + "\n" + self.clean_data(item["abstract"])

  def read_data(self):
    data = []
    # read line by line
    with open(self.path, "r") as f:

        for line in f:
            item = json.loads(line)

            text = self.preprocessing(item)
            data.append(text)
            #if len(data) >= 1000:
            #  break
    return data

  def visualize(self, path):
    data = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)

            data.append({
                "id": item["id"],
                "title": item["title"],
                "abstract": item["abstract"],
                "categories": item["categories"]
            })

            if i == 1000:  # solo 1000 papers
                break

    df = pd.DataFrame(data)

    display(df.head(10))
    display(df.isnull().sum())
    display(df["abstract"].sample(10).values)