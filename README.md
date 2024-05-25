# Exploring the World of Multimodal Models: An In-Depth Look at LLaVA

In the ever-evolving field of artificial intelligence, the fusion of various modalities—such as text, images, and audio—has paved the way for the development of powerful multimodal models. These models aim to understand and process information from multiple sources, mirroring the way humans perceive the world. In this blog, we will delve into the fundamentals of multimodal models, their origins, architecture, and functionalities. We will also explore the LLaVA model (Large Language and Vision Assistant), its significance, and an example of its application using Ollama.

## The Basis of Multimodal Models

### Origins and Evolution

The concept of multimodal models originates from the need to create AI systems that can interpret and generate information across different types of data. Traditionally, AI models were designed to handle specific tasks within a single modality—such as text generation or image recognition. However, real-world applications often require a more holistic understanding, combining multiple types of data to provide richer and more accurate outputs.

### Architecture and Functionality

Multimodal models integrate various neural network architectures to process and fuse data from different modalities. The core components typically include:

1. **Encoders**: Separate encoders for each modality (e.g., text, images) extract meaningful features from raw data.
2. **Fusion Mechanism**: This combines the extracted features into a unified representation, facilitating cross-modal interactions.
3. **Decoders**: Generate outputs based on the fused representation, tailored to the desired application (e.g., text descriptions, image captions).

A prominent example of a multimodal model is the LLaVA (Large Language and Vision Assistant), which exemplifies the integration of visual and language understanding in AI systems.

## Introducing LLaVA: Large Language and Vision Assistant

### Overview

LLaVA is an end-to-end trained large multimodal model designed to seamlessly combine a vision encoder and a language model for general-purpose visual and language understanding. This model stands out for its impressive chat capabilities, mimicking the multimodal prowess of advanced systems like GPT-4.

### Key Features

- **State-of-the-Art Performance**: LLaVA achieves new benchmarks in Science QA with 92.53% accuracy, surpassing models trained on billion-scale datasets.
- **Efficient Training**: Utilizing public data and completing training in approximately one day on a single 8-A100 node.
- **Open-Source Accessibility**: LLaVA and its associated data are publicly available, promoting transparency and further research in the field.

### Architecture

LLaVA connects a pre-trained CLIP ViT-L/14 visual encoder and a large language model, Vicuna, through a simple projection matrix. The training process is divided into two stages:

1. **Pre-training for Feature Alignment**: Only the projection matrix is updated using a subset of the CC3M dataset.
2. **Fine-tuning End-to-End**: Both the projection matrix and the language model are fine-tuned for specific use cases:
   - **Visual Chat**: Fine-tuning on multimodal instruction-following data for user-oriented applications.
   - **Science QA**: Fine-tuning on a multimodal reasoning dataset for the science domain.

## How LLaVA Works

LLaVA processes visual and textual data simultaneously, enabling it to generate contextually relevant responses based on visual inputs. For instance, given an image and a related query, LLaVA can describe the image, answer questions about its content, and even engage in complex reasoning tasks.

## Installing and Using Ollama to Run LLaVA

Ollama is a lightweight framework that allows for building and running language models on a local machine. Here's a step-by-step guide on how to install Ollama, load the LLaVA model, and run examples.

### Installation

1. **Install Ollama**:
   ```sh
   !curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start Ollama Service**:
   ```sh
   !rm "/content/myscript.sh"
   !echo '#!/bin/bash' > myscript.sh
   !echo 'ollama serve' >> myscript.sh
   !chmod +x myscript.sh
   !nohup ./myscript.sh &
   ```

3. **Run a Model**:
   ```sh
   !nohup ollama run phi3:medium > ollama_model.log 2>&1 &
   ```

4. **Pull the LLaVA Model**:
   ```sh
   !ollama pull llama3
   ```

5. **List Available Models**:
   ```sh
   !ollama list
   ```

## Example Use Case with Ollama
To illustrate LLaVA's capabilities, let's consider an example using Ollama, a lightweight framework for running language models locally.

Scenario: Describing an Image
Suppose we have an image of a famous painting and we want to identify the artist and provide details about the artwork.

Image Input: A picture of the Mona Lisa.

User Query: "Do you know who drew this painting?"

LLaVA Response:
The painting depicts a woman, commonly believed to be Mona Lisa, the famous artwork by Leonardo da Vinci. It is a portrait painting that showcases the woman's enigmatic smile and has become one of the most famous and iconic art pieces in the world. The original work is displayed in the Louvre Museum in Paris, and it is known for its intricate details, use of oil paint, and the artist's innovative techniques that contributed to its enduring appeal and mystery.

Running the Example with Ollama

Using Ollama, we can execute the following command to leverage LLaVA's capabilities:


$ ollama run llama3 "Summarize this file: $(cat README.md)"
Ollama provides a simple API to run models locally, enabling efficient and extensible applications of multimodal AI systems like LLaVA.


### Setting Up Python Environment

1. **Install Required Python Packages**:
   ```sh
   pip install -U langchain-community
   pip install embedchain
   pip install sentence-transformers
   pip install ollama
   pip install pyngrok
   ```

2. **Set Up Embedchain**:
   ```python
   from embedchain import App
   app = App.from_config(config={
       "llm": {
           "provider": "ollama",
           "config": {
               "model": "llama3",
               "temperature": 0.5,
               "top_p": 1,
               "stream": True
           }
       },
       "embedder": {
           "provider": "huggingface",
           "config": {
               "model": "BAAI/bge-small-en-v1.5"
           }
       }
   })
   ```

3. **Run a Query**:
   ```python
   answer = app.query("who is elon musk?")
   print(answer)
   ```

### Example: Extracting Medical Intervention Names

Here's a Python function to extract medical intervention names from clinical trial descriptions using the LLaVA model with Ollama:

```python
import ollama

def extractDrugs(prompt: str, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is to extract medical intervention names (drugs or therapies) from the details of a clinical trial. Each trial can have multiple interventions."
        "Give a json list of intevention names with the following information for each: name, type, is_new, conditions, list of trial ids, dosage & administration"
        "Separate the dosage & administration, keep intervention names concise, only including identifying information about the drug or therapy."
        "The response should have the following format:"
        '['
        '   {'
        '       "name": Intervention name,'
        '       "type": Biologic,'
        '       "is_new_drug": Yes (if its a new drug),'
        '       "conditions": ["condition1", "condition2"],'
        '       "dosage_administration": {'
        '           "dosage": "0.02%",'
        '           "administration": "topical, 4 times a day for 12 weeks"'
        '       }'
        "   },"
        "   {"
        '       "name": Intervention name 2,'
        '       "type": Small molecule'
        '       "is_new_drug": No (if its a generic, biosimilar or not mentioned),'
        '       "conditions": ["condition1", "condition2"],'
        '       "dosage_administration": {'
        '           "dosage": "0.02%",'
        '           "administration": "topical, 4 times a day for 12 weeks"'
        '       }'
        '   },'
        ']'
    )

    result = ollama.generate(
        model=model,
        prompt=prompt,
        system=SYS_PROMPT
    )
    print(result['response'])
    return result['response']
```

### Running the Example

To run the example, you can load XML data, process it to extract clinical trial information, and then use the `extractDrugs` function to identify drugs mentioned in the trials:

```python
import xml.etree.ElementTree as ET

TAGS_TO_KEEP = [
    'Internal_Number',
    'TrialID',
    'Public_title',
    'Scientific_title',
    'Condition',
    'Intervention'
]

class TrialXMLLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        dict = self._structuredXMLLoader()
        trials = []
        for elem in dict.get('children', []):
            trials.append(','.join([
                f"""(tag: {x.get('tag','').strip()}, text: {x.get('text','').strip()})"""
                for x in elem.get('children', [])
                if x.get('text','') and x.get('tag','') in TAGS_TO_KEEP
            ]))
        return trials

    def _structuredXMLLoader(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        return self._parse_element(root)

    def _parse_element(self, element):
        return {
            "tag": element.tag,
            "text": element.text.strip() if element.text else None,
            "attributes": element.attrib,
            "children": [self._parse_element(child) for child in element]
        }

xml_path = "./xml-1.xml"
loader = TrialXMLLoader(file_path=xml_path)

documents = loader.load()
drugs_list = trials2DrugsList(documents[:10], model='phi3:medium')
print(drugs_list[3])
```

## Inference with LLaVA Without Ollama

You can also perform inferences with LLaVA without using Ollama. This involves directly using the Hugging Face Transformers library to load the LLaVA model and process inputs. Below is a detailed guide on setting up the environment

 and performing inferences with LLaVA.

### Setting Up the Environment

1. **Install Necessary Libraries**:
   ```sh
   !pip install --upgrade -q accelerate bitsandbytes
   !pip install git+https://github.com/huggingface/transformers.git
   ```

### Loading the Model and Processor

2. **Load Model and Processor**:
   ```python
   from transformers import AutoProcessor, LlavaForConditionalGeneration
   from transformers import BitsAndBytesConfig
   import torch

   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
   )

   model_id = "llava-hf/llava-1.5-7b-hf"

   processor = AutoProcessor.from_pretrained(model_id)
   model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
   ```

### Preparing Image and Text for the Model

3. **Prepare Image and Text**:
   ```python
   import requests
   from PIL import Image

   image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
   image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
   display(image1)
   display(image2)
   ```

### Generating Outputs

4. **Generate Text**:
   ```python
   prompts = [
               "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
               "USER: <image>\nPlease describe this image\nASSISTANT:",
   ]

   inputs = processor(prompts, images=[image1, image2], padding=True, return_tensors="pt").to("cuda")
   for k,v in inputs.items():
     print(k,v.shape)

   output = model.generate(**inputs, max_new_tokens=20)
   generated_text = processor.batch_decode(output, skip_special_tokens=True)
   for text in generated_text:
     print(text.split("ASSISTANT:")[-1])
   ```

### Using the Pipeline API

5. **Pipeline API**:
   ```python
   from transformers import pipeline

   pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

   max_new_tokens = 200
   prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place?\nASSISTANT:"

   outputs = pipe(image1, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

   print(outputs[0]["generated_text"])
   ```

## Conclusion

Multimodal models like LLaVA represent a significant leap forward in AI capabilities, enabling richer and more nuanced understanding of complex information by integrating visual and textual data. Whether using frameworks like Ollama or directly leveraging the Hugging Face library, developers can harness the power of these models to create sophisticated applications across various domains. By exploring and utilizing such advanced models, we continue to push the boundaries of what AI can achieve, bringing us closer to truly intelligent systems that can perceive and interact with the world in a human-like manner.