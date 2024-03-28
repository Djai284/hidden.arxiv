# hidden arXiv: Research Paper Processor

The Hidden arXiv Research Paper Processor is a Python script designed to automate the fetching, processing, and dissemination of research papers from arXiv. It streamlines the extraction of text from PDFs, generates summaries, converts these summaries to speech, overlays the audio on a template video, and uploads the final video content to platforms like YouTube as shorts. The script also manages the storage of processed content in AWS S3 and records processed papers in AWS DynamoDB.

## Features

- **Fetch Recent Papers**: Compiles a list of recent papers from specified arXiv categories.
- **PDF Processing**: Downloads PDFs and extracts text.
- **Cloud Storage**: Uploads extracted text and generated audio files to S3.
- **Content Generation**: Summarizes text using GPT and converts summaries to speech.
- **Video Production**: Overlays generated audio on template videos using FFmpeg.
- **YouTube Upload**: Automates the upload of videos as YouTube Shorts.
- **DynamoDB Integration**: Records the processing status of papers in DynamoDB.
- **Content Duplication Check**: Ensures papers are only processed once.

## Prerequisites

- Python 3.x
- `requests`, `beautifulsoup4`, `PyMuPDF (fitz)`, `boto3`, `openai`, `argparse`, and other Python libraries.
- AWS CLI configured with access to S3 and DynamoDB.
- API keys for OpenAI and Eleven Labs stored as environment variables.

## Installation

1. Clone this repository.
2. Install required Python packages:
```
pip install -r requirements.txt
```

3. Ensure AWS CLI is configured with the necessary permissions.
4. Set the environment variables for OpenAI and Eleven Labs API keys:
```
export OPENAI_API_KEY='your_openai_api_key'
export ELEVEN_LABS_API_KEY='your_eleven_labs_api_key'
```


## Usage

Run the script from the command line, specifying the subject and paper limit as arguments:

```
python arxiv_scraping.py --subject cs --paper-limit 10
```


Available subjects include `cs`, `physics`, `math`, `bio`, `finance`, `stats`, `ee`, and `econ`.

## Configuration

Edit the `source_links` dictionary to add or modify subjects and their corresponding arXiv URLs.

## AWS Configuration

The script uses `boto3` for AWS interactions. Ensure your AWS credentials are set up for programmatic access, and the necessary permissions are granted for S3 and DynamoDB.

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please submit pull requests or open issues to propose changes or additions.
