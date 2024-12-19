import express from "express";
import { fetchEmails, classifyEmails, validateToken } from "../utils/emailParser.js";

const router = express.Router();


// POST /api/emails
router.post("/", async (req, res) => {
  const { token } = req.body;

  if (!token) {
    return res
      .status(400)
      .json({ error: "Token is required in the request body" });
  }

  try {
    // Validate token before proceeding
    await validateToken(token);

    console.log("Token validation successful. Fetching emails...");
    const emails = await fetchEmails(token);

    if (!emails || emails.length === 0) {
      return res
        .status(404)
        .json({ error: "No emails found for the given token" });
    }

    console.log("Emails fetched successfully. Classifying emails...");
    const categorizedEmails = await classifyEmails(emails);

    // Respond with the categorized emails
    console.log("Categorized Emails Response:", categorizedEmails);
    res.status(200).json({ success: true, categorizedEmails });
  } catch (error) {
    console.error("Error processing emails:", error);

    // Respond with an appropriate error message
    const errorMessage = error.message.includes("Token validation failed")
      ? "Invalid or expired token."
      : error.message.includes("Gmail API")
      ? "Failed to fetch emails from Gmail API"
      : "Internal server error while classifying emails";

    res.status(500).json({ error: errorMessage });
  }
});

export default router;
