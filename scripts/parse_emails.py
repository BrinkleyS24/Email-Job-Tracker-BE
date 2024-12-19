import mailbox
import json

def parse_mbox_to_json(mbox_file, output_json):
    mbox = mailbox.mbox(mbox_file)
    with open(output_json, 'w', encoding='utf-8') as f:
        f.write('[')  # Start of JSON array
        first = True

        for msg in mbox:
            try:
                # Extract body content safely
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode(errors="ignore")
                                break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode(errors="ignore")

                email_data = {
                    "subject": msg["subject"] or "",
                    "body": body,
                    "from": msg["from"] or "",
                    "date": msg["date"] or ""
                }

                if not first:
                    f.write(',\n')
                f.write(json.dumps(email_data))
                first = False
            except Exception as e:
                print(f"Error processing message: {e}")

        f.write(']')  # End of JSON array
    
    mbox.close()

# Usage
parse_mbox_to_json("smaller_inbox.mbox", "parsed_emails.json")
