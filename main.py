import json
import os
import chromadb
import imaplib
import smtplib
import email
from email.message import EmailMessage
from email.header import decode_header

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

# NEW: Load email credentials and server details from environment variables
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))


# ─────────────────────────────────────────────
# 1. Load knowledge base from JSON
# ─────────────────────────────────────────────

def load_knowledge_base(filepath="knowledge_base.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


# ─────────────────────────────────────────────
# 2. Insert documents into ChromaDB
# ─────────────────────────────────────────────

def build_chroma_collection(knowledge_base):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    chroma_client = chromadb.Client()

    vectorstore = Chroma(
        collection_name="adib_knowledge",
        embedding_function=embedding_function,
        client=chroma_client,
    )

    documents = []
    for item in knowledge_base:
        doc = Document(
            page_content=item["content"],
            metadata={"title": item["title"], "id": item["id"]}
        )
        documents.append(doc)

    vectorstore.add_documents(documents)

    return vectorstore


# ─────────────────────────────────────────────
# 3. Extract fields using Groq function calling
# ─────────────────────────────────────────────

def extract_email_fields(email_text):
    client = OpenAI(api_key=GROQ_API_KEY, base_url=BASE_URL)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_customer_info",
                "description": "Extract key fields from a customer banking support email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_name": {
                            "type": "string",
                            "description": "Full name of the customer. Empty string if not found."
                        },
                        "request_type": {
                            "type": "string",
                            "description": "Type of request the customer is making, e.g. card activation, cashback inquiry. Empty string if not found."
                        },
                        "card_type": {
                            "type": "string",
                            "description": "Type of card mentioned, e.g. debit, credit, prepaid, cashback. Empty string if not found."
                        },
                        "priority": {
                            "type": "string",
                            "description": "Priority level: high, medium, or low. Guess based on urgency in the email. Empty string if not found."
                        }
                    },
                    "required": ["customer_name", "request_type", "card_type", "priority"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a banking support assistant. Extract information from customer emails."
        },
        {
            "role": "user",
            "content": f"Extract the required fields from this customer email:\n\n{email_text}"
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_customer_info"}}
    )

    tool_call = response.choices[0].message.tool_calls[0]
    extracted = json.loads(tool_call.function.arguments)

    return extracted


# ─────────────────────────────────────────────
# 4. Search ChromaDB using request_type
# ─────────────────────────────────────────────

def search_knowledge(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    return results


# ─────────────────────────────────────────────
# 4.5. Simple Agent Implementation
# ─────────────────────────────────────────────

class SimpleAgent:
    def __init__(self, role, goal, backstory, llm):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm

    def run(self, task_description):
        prompt = f"Role: {self.role}\nGoal: {self.goal}\nBackstory: {self.backstory}\n\nTask:\n{task_description}"
        response = self.llm.invoke(prompt)
        return response.content

agent_llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    base_url=BASE_URL
)

researcher = SimpleAgent(
    role="Credit Card Policy Researcher",
    goal="Summarize the exact card policy that applies to the customer request in 1–2 sentences.",
    backstory="You are meticulous and only state what exists inside the retrieved banking policy. Never invent information.",
    llm=agent_llm
)

def run_researcher(extracted_data, retrieved_policy, original_email):
    extracted_json = json.dumps(extracted_data)
    
    policy_text = ""
    for i, doc in enumerate(retrieved_policy, start=1):
        title = doc.metadata.get("title", "Unknown")
        policy_text += f"\nDocument {i} - {title}:\n{doc.page_content}\n"
        
    task_description = f"Customer Email:\n{original_email}\n\nExtracted Data:\n{extracted_json}\n\nRetrieved Policy:\n{policy_text}\n"
    
    response = researcher.run(task_description)
    return response


reviewer = SimpleAgent(
    role="Quality Reviewer",
    goal="Review the drafted email response and make sure it is professional, accurate, and complete.",
    backstory="You are a senior banking support reviewer. You check that the response is polite, addresses the customer by name, and does not contain made-up information.",
    llm=agent_llm
)

def run_reviewer(resolver_output, original_email):
    task_description = f"Original Customer Email:\n{original_email}\n\nDrafted Response:\n{resolver_output}\n\nPlease review and return the final polished email response."
    response = reviewer.run(task_description)
    return response


# ─────────────────────────────────────────────
# 5. Resolver Agent - one prompt, one response
# ─────────────────────────────────────────────

def resolver_agent(email_text, extracted_fields, researcher_summary=None):
    llm = ChatOpenAI(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        base_url=BASE_URL
    )

    if researcher_summary:
        context_section = f"Researcher Summary:\n{researcher_summary}"
    else:
        context_section = "No policy documents were retrieved for this request type."

    prompt = f"""You are a professional customer support agent for ADIB (Abu Dhabi Islamic Bank).

You received the following customer email:
---
{email_text}
---

Extracted customer information:
{json.dumps(extracted_fields, indent=2)}

Relevant knowledge base documents:
{context_section}

Instructions:
- Write a professional and helpful banking support response.
- Use ONLY the information from the retrieved documents above.
- Do NOT make up any information or policies not mentioned in the documents.
- Address the customer by name if available.
- Keep the tone polite, clear, and professional.
- Sign off as: ADIB Customer Support Team

Write the outgoing email response now:
"""

    response = llm.invoke(prompt)
    return response.content


# ─────────────────────────────────────────────
# 6. Email Operations (IMAP/SMTP)
# ─────────────────────────────────────────────

def fetch_unread_emails():
    # NEW: Fetch unread emails from the IMAP server
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("Email credentials not set. Cannot fetch emails.")
        return []

    try:
        # NEW: Connect to IMAP server using SSL
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select('inbox')

        status, response = mail.search(None, 'UNSEEN')
        unread_email_ids = response[0].split()
        
        emails = []
        for num in unread_email_ids:
            status, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            
            # Decode subject
            subject, encoding = decode_header(msg['subject'])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else 'utf-8')
                
            sender = msg['from']
            body = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors='ignore')
                        break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors='ignore')
                
            emails.append({
                'id': num,
                'subject': subject,
                'sender': sender,
                'body': body,
                'mail_client': mail
            })
            
        return emails
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def mark_email_as_read(mail_client, email_id):
    # NEW: Mark an email as read on the IMAP server so it is not processed again
    try:
        mail_client.store(email_id, '+FLAGS', '\\Seen')
    except Exception as e:
        print(f"Error marking email as read: {e}")

def send_email_reply(to_address, subject, body):
    # NEW: Send an email reply using the SMTP server via SSL (Port 465)
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("Email credentials not set. Cannot send reply.")
        return

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = f"Re: {subject}"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_address

        # NEW: Use SMTP_SSL instead of starttls() for port 465
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Reply successfully sent to {to_address}")
    except Exception as e:
        print(f"Error sending email reply: {e}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # Load and index knowledge base
    knowledge_base = load_knowledge_base()
    vectorstore = build_chroma_collection(knowledge_base)

    # Print document count
    print(f"Documents in ChromaDB collection: {vectorstore._collection.count()}\n")

    print("Checking for unread emails...")
    # NEW: Fetch unread emails from an email inbox (IMAP).
    unread_emails = fetch_unread_emails()
    
    if not unread_emails:
        print("No new emails found.")
        return

    # NEW: Loop through each unread email for processing
    for email_info in unread_emails:
        # NEW: Read sender email, subject, and body
        email_id = email_info["id"]
        sender = email_info["sender"]
        subject = email_info["subject"]
        email_text = email_info["body"]
        mail_client = email_info["mail_client"]

        print("=" * 40)
        print(f"========== Incoming Email ==========")
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print("Body:")
        print(email_text)

        # NEW: Pass the email body to the existing extract_email_fields() function
        extracted_fields = extract_email_fields(email_text)

        print("\n========== Extracted Fields ==========")
        print(json.dumps(extracted_fields, indent=2))

        # NEW: Determine the request type
        request_type = extracted_fields.get("request_type", "").lower()

        # NEW: Route flow if the request type is an Inquiry
        if "inquiry" in request_type or "inquir" in request_type:
            print("\n[Route: Inquiry]")

            search_query = extracted_fields.get("request_type", email_text)
            retrieved_docs = search_knowledge(vectorstore, search_query)

            print("\n========== Retrieved Documents ==========")
            for i, doc in enumerate(retrieved_docs, start=1):
                title = doc.metadata.get("title", "Unknown")
                print(f"\n[{i}] {title}")
                print(doc.page_content)

            researcher_summary = run_researcher(extracted_fields, retrieved_docs, email_text)

            print("\n========== Researcher Summary ==========")
            print(researcher_summary)

            resolver_output = resolver_agent(email_text, extracted_fields, researcher_summary)

        # NEW: Otherwise, route flow for Non-Inquiry (Skip Knowledge Base)
        else:
            print("\n[Route: Non-Inquiry]")
            resolver_output = resolver_agent(email_text, extracted_fields)

        print("\n========== Resolver Response ==========")
        print(resolver_output)

        final_email = run_reviewer(resolver_output, email_text)

        print("\n========== Reviewer Output ==========")
        print(final_email)

        print("\n========== OUTGOING EMAIL ==========")
        print(final_email)
        
        # NEW: Send the final response back to the original sender using SMTP
        print("\nSending email reply...")
        send_email_reply(sender, subject, final_email)
        
        # NEW: Mark the processed email as Read
        print("Marking email as read...")
        mark_email_as_read(mail_client, email_id)
        
        # NEW: Continue processing the next unread email
        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
