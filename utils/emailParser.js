import { spawn } from "child_process";
import fetch from "node-fetch";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Validate the token before fetching emails
const validateToken = async (token) => {
  const url = `https://www.googleapis.com/oauth2/v1/tokeninfo?access_token=${token}`;
  const response = await fetch(url);

  if (!response.ok) {
    console.error("Token validation failed:", response.statusText);
    throw new Error("Token validation failed.");
  }

  const data = await response.json();
  return data;
};

const JOB_EMAIL_QUERY = {
  include: [
    "application has been received",
    "thank you for applying",
    "application confirmation",
    "schedule an interview",
    "interview invitation",
    "job offer",
    "congratulations",
    "excited to offer you",
    "we regret to inform you",
    "not moving forward",
    "not a fit",
    "your application",
    "unfortunately has decided",
    "not to move forward",
    "Take-Home Exercise",
    "assessment",
    "next steps",
    "we are reviewing",
    "interview feedback",
    "offer letter",
    "response to your application",
    "interview",
    "Take-Home",
    "Thanks for your interest",
    "Microsoft Teams"
  ],
  exclude: [
    "offer",
    "membership",
    "deal",
    "discount",
    "subscribe",
    "Job alert",
    "quora digest",
    "promotion",
    "sale",
    "upgrade",
    "advice",
    "digest",
    "event invite",
    "newsletter",
    "password reset",
    "internship",
    "career tips",
    "job search tips",
    "referral program",
    "webinar",
    "free resources",
    "recruitment fair",
    "event update",
    "employee referral",
    "networking event",
    "Tell us how your assessment went"
  ],
  timeframe: "newer_than:30d",
};

const fetchEmails = async (token) => {
  if (process.env.USE_MOCK_DATA === "true") {
    console.log("Using mock data instead of fetching from Gmail API...");
    return [
      {
        "subject": "We’ve Received Your Application!",
        "from": "careers@company1.com",
        "body": "Thank you for your application to the Software Engineer role. We’ll review your credentials and get back soon.",
        "date": "2024-12-01T10:00:00Z"  // Applied
      },
      {
        "subject": "Can You Confirm Your Interview Slot?",
        "from": "hr@company2.com",
        "body": "We noticed you haven’t confirmed your interview slot for the position. Please reply to confirm by tomorrow.",
        "date": "2024-12-02T12:00:00Z"  // Interviewed
      },
      {
        "subject": "Great News – A Role for You!",
        "from": "offers@company3.com",
        "body": "We’re excited to offer you the Data Analyst position. Check the attached document for details.",
        "date": "2024-12-03T14:00:00Z"  // Offers
      },
      {
        "subject": "Status Update Regarding Your Application",
        "from": "notifications@company4.com",
        "body": "We appreciate your interest but regret to inform you that we’ve moved forward with another candidate.",
        "date": "2024-12-04T16:00:00Z"  // Rejected
      },
      {
        "subject": "Exclusive Career Coaching Offer – Limited Time Only!",
        "from": "promotions@company5.com",
        "body": "Achieve your career goals with our coaching services. 50% off this week only!",
        "date": "2024-12-05T11:00:00Z"  // Irrelevant
      },
      {
        "subject": "Follow-up on Your Job Application",
        "from": "hiring@company6.com",
        "body": "Hello, just following up on your recent job application. Are you still interested in the role?",
        "date": "2024-12-06T13:00:00Z"  // Applied
      },
      {
        "subject": "We’d Like to Discuss an Offer",
        "from": "team@company7.com",
        "body": "Congratulations on reaching the final stage! We’d love to discuss an offer with you. Please let us know your availability.",
        "date": "2024-12-07T15:00:00Z"  // Offers
      },
      {
        "subject": "Interview Cancelled - Please Reschedule",
        "from": "recruitment@company8.com",
        "body": "Your interview for the Product Manager role has been canceled. Reach out to reschedule if you’re still interested.",
        "date": "2024-12-08T17:00:00Z"  // Interviewed
      },
      {
        "subject": "Thanks for Applying – We’ll Be in Touch",
        "from": "applications@company9.com",
        "body": "We’ve received your resume and will contact you shortly if selected for the next steps.",
        "date": "2024-12-09T10:00:00Z"  // Applied
      },
      {
        "subject": "Re: Application Status Inquiry",
        "from": "jobs@company10.com",
        "body": "Thank you for checking in on your application status. At this time, we have no updates to share.",
        "date": "2024-12-10T12:00:00Z"  // Ambiguous: Could be Applied or Rejected
      }
    ]
    

  }

  const query = `in:inbox (${JOB_EMAIL_QUERY.include
    .map((term) => `"${term}"`)
    .join(" OR ")}) -from:me -{${JOB_EMAIL_QUERY.exclude
      .map((term) => `"${term}"`)
      .join(" OR ")}} ${JOB_EMAIL_QUERY.timeframe}`;


  const url = `https://gmail.googleapis.com/gmail/v1/users/me/messages?q=${encodeURIComponent(query)}&maxResults=100`;

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    const errorClone = response.clone();
    const errorText = await errorClone.text();
    console.error(`Error fetching emails: ${response.status}, ${errorText}`);
    throw new Error("Failed to fetch emails from Gmail API");
  }

  const data = await response.json();

  if (!data.messages) {
    return [];
  }

  return fetchEmailDetails(data.messages, token);
};


// Fetch details for each email with batching
const fetchEmailDetails = async (messages, token) => {
  const MAX_PARALLEL_REQUESTS = 5;
  const MAX_RETRIES = 3;

  const fetchDetail = async (msg) => {
    let attempts = 0;

    while (attempts < MAX_RETRIES) {
      try {
        const detailUrl = `https://gmail.googleapis.com/gmail/v1/users/me/messages/${msg.id}`;
        const response = await fetch(detailUrl, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!response.ok) {
          if (response.status === 429) {
            // Handle rate limit
            const retryAfter = response.headers.get("Retry-After");
            const delay = (retryAfter ? parseInt(retryAfter, 10) : 1) * 1000;
            console.error(
              `Rate limit exceeded for email ID ${msg.id}. Retrying after ${delay / 1000} seconds...`
            );
            await sleep(delay);
            continue;
          }

          const errorText = await response.text();
          console.error(
            `Failed to fetch email details for ID ${msg.id}. Status: ${response.status}, Response: ${errorText}`
          );
          return null; // Skip on failure
        }

        // Parse response
        const email = await response.json();
        const bodyData =
          email.payload.body?.data ||
          email.payload.parts?.find((part) => part.mimeType === "text/plain")
            ?.body?.data ||
          "";

        return {
          subject:
            email.payload.headers.find((header) => header.name === "Subject")
              ?.value || "",
          body: Buffer.from(bodyData, "base64").toString("utf-8"),
          from:
            email.payload.headers.find((header) => header.name === "From")
              ?.value || "",
          date:
            email.payload.headers.find((header) => header.name === "Date")
              ?.value || "",
        };
      } catch (error) {
        attempts++;

        if (error.code === "ERR_STREAM_PREMATURE_CLOSE") {
          console.error(
            `Stream prematurely closed for email ID ${msg.id}, attempt ${attempts}. Retrying...`
          );
        } else {
          console.error(
            `Error processing email with ID ${msg.id}, attempt ${attempts}: ${error.message}`
          );
        }

        if (attempts >= MAX_RETRIES) {
          console.error(
            `Max retries reached for email ID ${msg.id}. Skipping...`
          );
          return null;
        }

        await sleep(1000 * attempts); // Exponential backoff
      }
    }
  };

  const batches = [];
  for (let i = 0; i < messages.length; i += MAX_PARALLEL_REQUESTS) {
    const batch = messages
      .slice(i, i + MAX_PARALLEL_REQUESTS)
      .map((msg) => fetchDetail(msg));
    const results = await Promise.all(batch);
    batches.push(...results);
  }
  return batches.filter(Boolean);
};


// Classify emails using a Python script
const classifyEmails = async (emails) => {
  const results = [];
  const categories = { Applied: [], Interviewed: [], Offers: [], Rejected: [], Irrelevant: [] };

  for (let i = 0; i < emails.length; i += 2) { // Small batch size
      const batch = emails.slice(i, i + 2);

      try {
          const result = await new Promise((resolve, reject) => {
              const pythonProcess = spawn("python", [
                  "C:\\Users\\stace\\gmail-job-tracker-be\\scripts\\classify_email.py",
              ]);

              let stdout = "";
              let stderr = "";

              pythonProcess.stdin.write(JSON.stringify(batch));
              pythonProcess.stdin.end();

              pythonProcess.stdout.on("data", (data) => {
                  stdout += data.toString();
              });

              pythonProcess.stderr.on("data", (data) => {
                  stderr += data.toString();
              });

              pythonProcess.on("close", (code) => {
                  if (code !== 0) {
                      reject(new Error(`Python script failed with code ${code}: ${stderr}`));
                  } else {
                      try {
                          const parsed = JSON.parse(stdout);
                          if (!Array.isArray(parsed)) {
                              throw new Error("Parsed output is not an array.");
                          }
                          resolve(parsed);
                      } catch (err) {
                          reject(new Error(`Failed to parse Python output: ${err.message}`));
                      }
                  }
              });

              pythonProcess.on("error", reject);
          });

          results.push(...result);
      } catch (err) {
          console.error(`Error classifying batch: ${err.message}`);
      }
  }

  results.forEach((email) => {
      const category = email.category;
      if (category && categories[category]) {
          categories[category].push(email);
      } else {
          categories.Irrelevant.push(email);
      }
  });

  return categories;
};



export { fetchEmails, classifyEmails, validateToken };
