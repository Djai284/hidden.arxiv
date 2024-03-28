import argparse
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import re
import boto3, botocore
import os
import datetime
import openai
from textwrap import wrap
from prompts import user_prompt, ai_response


import concurrent.futures
from functools import partial


'''
The goal of this script is to fetch the PDFs from arxiv.
1. compile a list of links to scrape papers off of
2. scrape arxiv link for recent papers posted to category
3. filter out pdfs that have already been processed
3. download pdf
4. extract text from pdf
5. upload text file to S3
6. mark specific paper as processed in DynamoDB
7. delete pdf
8. upload text file to Claude and get summary and upload to S3 in the same folder
9. send summary to eleven labs -> retrieve audio file
10. upload audio file to S3 in the same folder as the text file
11. fetch template video from repository (story blocks) 
12. use ffmpeg to overlay audio with video and generate a new video file
13. store final video in S3 in the same folder
14. use Youtube API to upload video as a short
'''

# Take in CLI arguments to 
def parse_args():
    parser = argparse.ArgumentParser(description="Get scenes from a PDF")
    
    parser.add_argument(
      "--subject", 
      default="cs", 
      choices=source_links.keys(),
      help="the research subject to query"
    )
    
    parser.add_argument(
      '--paper-limit', 
      type=int, 
      help='Number of papers to retrieve', 
      default=10
    )

    args = parser.parse_args()
    return args

# Define your AWS S3 bucket name
S3_BUCKET_NAME = 'hidden-arxiv-research-data'

# Configure your AWS credentials (preferably in your environment variables or AWS credentials file)
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize the S3 client
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Initializing the DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table("arxiv-processed-papers")

# Initializing GPT Client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Eleven Labs API Key
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")

# 1. List of links to fetch PDFs from
source_links = {
  "cs": "https://arxiv.org/list/cs/recent",
  "physics": "https://arxiv.org/list/physics/recent",
  "math": "https://arxiv.org/list/math/recent",
  "bio": "https://arxiv.org/list/q-bio/recent",
  "finance": "https://arxiv.org/list/q-fin/recent",
  "stats": "https://arxiv.org/list/stat/recent",
  "ee": "https://arxiv.org/list/eess/recent",
  "econ": "https://arxiv.org/list/econ/recent"
}

# 2. Scrape arxiv links 
def fetch_arxiv_pdf_links(link, num_articles=1):
    url = link
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find <a> tags with 'pdf' in the href attribute, limited by num_articles
    pdf_links = soup.find_all('a', title='Download PDF', limit=num_articles)
    pdf_urls = ['https://arxiv.org' + link['href'] for link in pdf_links]

    return pdf_urls

# 3. Check if paper has already been processed
def is_paper_processed(arxiv_id):
    """
    Checks if the paper with the given arXiv ID has already been processed.
    
    Args:
    arxiv_id (str): The arXiv ID of the paper.
    
    Returns:
    bool: True if the paper has already been processed, False otherwise.
    """
    try:
      response = table.get_item(Key={'arxiv-id': arxiv_id})
      return 'Item' in response
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"Table not found: Check that the table exists and that your AWS region is correct.")
        else:
            print(f"Unexpected error: {error.response['Error']['Message']}")
        return False
 
# 4. Download PDF
def download_pdf(url, filename):
    """Download PDF from a given URL and save it as filename."""
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# 5. Extracting text from PDF through OCR     
def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF and return it."""
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# 6. Saving txt file in S3
def extract_arxiv_id_from_url(pdf_url):
    """
    Extracts the arXiv ID from the given PDF URL.

    Args:
    pdf_url (str): The URL of the arXiv PDF.

    Returns:
    str: The extracted arXiv ID, or None if the pattern does not match.
    """
    # Regex pattern to match the arXiv ID format
    # This pattern accounts for the common format starting with year and month
    pattern = r'arxiv\.org\/pdf\/(\d{4}\.\d{4,5}(v\d+)?)(?:\.pdf)?'
    match = re.search(pattern, pdf_url)
    
    if match:
        return match.group(1)
    else:
        return None

def save_text_to_s3(text, filename):
    # Convert text to bytes
    text_bytes = text.encode('utf-8')
    # Upload directly to S3 without saving to disk
    s3_client.put_object(Body=text_bytes, Bucket=S3_BUCKET_NAME, Key=filename)

# 7. Marking paper as processed in DynamoDB
def mark_paper_as_processed(arxiv_id):
    """
    Mark the paper as processed in DynamoDB.
    
    Args:
    arxiv_id (str): The arXiv ID of the paper.
    """
    now = datetime.datetime.now()
    response = table.put_item(
       Item={
            'arxiv-id': arxiv_id,
            'ProcessedTimestamp': now.isoformat(),
            'Status': 'Processed'
        }
    )
    return response

# 8: Uploading text to GPT and get result back
def summarize_text(text, model="gpt-4-0125-preview", max_tokens=4000):
    """
    Summarize a long text using GPT-3.5 Turbo by breaking it into segments.
    
    Args:
    text (str): The long text to summarize.
    model (str): The model to use for summarization.
    max_tokens (int): Maximum number of tokens that can be processed by the model in a single request.
    
    Returns:
    str: The summarized text.
    """
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
          {"role": "system", "content": "You are a helpful assistant. Your goal is to summarize research papers so that a general audience can understand it's importance in the field. Make it concise so that it can fit within a 50 second video clip "},
          {"role": "user", "content": user_prompt},
          {"role": "assistant", "content": ai_response},
          {"role": "user", "content": f"Summarize the paper below. Capture the audience's attention in the beginning with a question or bold statement about how the research presented could affect the field or people's lives. Then go into the specifics of the research. Don't make it sound like an advertisement:\n {text}"}
        ],
        max_tokens=175
    )
    summary = response.choices[0].message.content
    
    return summary

# 9: Send summary to 11 Labs to get the audio file
def text_to_speech(summary, arxiv_id):

  CHUNK_SIZE = 1024
  voice_id = "29vD33N1CtxCmqQRPOHJ"
  url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

  headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": eleven_labs_api_key
  }

  data = {
    "text": summary,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
      "stability": 0.5,
      "similarity_boost": 0.5
    }
  }

  response = requests.post(url, json=data, headers=headers)
  mp3_filename = f'{arxiv_id}.mp3'
  with open(mp3_filename, 'wb') as f:
      for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
          if chunk:
              f.write(chunk)
              
  return mp3_filename

# Steps to process a single paper
def process_paper(url, subject):
    print(f"Processing: {url}")
    arxiv_id = extract_arxiv_id_from_url(url)
    if arxiv_id:
        # Placeholder for is_paper_processed check
        pdf_filename = f'{arxiv_id}.pdf'
        txt_filename = f'{arxiv_id}.txt'
        partition = f"arxiv/{subject}/{arxiv_id}/"
        
        download_pdf(url, pdf_filename)
        text = extract_text_from_pdf(pdf_filename)
        
        # Saving the extracted text in S3
        save_text_to_s3(text, partition + txt_filename)
        
        # Placeholder for GPT summary generation and saving to S3
        summary = summarize_text(text)
        print(f"Summary for {arxiv_id}: {summary[:100]}...")  # Print a snippet of the summary
        summary_filename = f'{arxiv_id}-summary.txt'
        save_text_to_s3(summary, partition + summary_filename)
        
        # Placeholder for marking paper as processed
        mark_paper_as_processed(arxiv_id)
        
        # Clean up by deleting the PDF file after processing
        os.remove(pdf_filename)
        
        print(f'Processed and uploaded: {txt_filename}')
        
        # Produce the audio file for the TTS
        mp3_filename = text_to_speech(summary, arxiv_id=arxiv_id)
        
        # Read mp3 file and upload to s3 in the same folder
        try:
          s3_client.upload_file(mp3_filename, S3_BUCKET_NAME, partition + mp3_filename,
                                ExtraArgs={'ContentType': 'audio/mp3'})
        except Exception as e:
            print(f"An error occurred: {e}")
          
        # cleanup by deleting the mp3 file after uploading
        # os.remove(mp3_filename)
        
    else:
        print("Could not extract arXiv ID from URL")

# Main Driver function     
def main():
    args = parse_args()
    pdf_urls = fetch_arxiv_pdf_links(source_links[args.subject], args.paper_limit)
    
    args = parse_args()
    pdf_urls = fetch_arxiv_pdf_links(source_links[args.subject], args.paper_limit)
    
    # Partial function for process_paper with the subject filled in
    process_with_subject = partial(process_paper, subject=args.subject)
    
    # Use ThreadPoolExecutor to process papers in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_with_subject, url) for url in pdf_urls]
        
        # Wait for all futures to complete (optional, if you want to handle results/errors)
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Raises exception of the function if occurred
            except Exception as e:
                print(f"Error processing paper: {e}")
        
if __name__ == '__main__':
    print('fetching PDF urls')
    main()