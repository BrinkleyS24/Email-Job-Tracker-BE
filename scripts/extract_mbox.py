import mailbox

def extract_emails(input_mbox, output_mbox, max_emails=1000):
    # Open the input mbox file
    mbox = mailbox.mbox(input_mbox)
    
    # Open or create the output mbox file
    outbox = mailbox.mbox(output_mbox)
    
    # Clear any existing content in the output mbox
    outbox.lock()  # Lock the file for thread safety
    outbox.clear()  # Remove any existing emails in the output file
    outbox.flush()
    outbox.unlock()
    
    # Add emails to the output mbox file
    for i, msg in enumerate(mbox):
        if i >= max_emails:
            break
        outbox.add(msg)
    
    # Save changes to the output mbox
    outbox.flush()
    outbox.close()
    mbox.close()

# Usage
extract_emails("../Inbox-001.mbox", "smaller_inbox.mbox", max_emails=1000)
