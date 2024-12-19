import emailRoutes from "./routes/emailRoutes.js";
import express from "express";
import bodyParser from "body-parser";
import cors from "cors";

const app = express();

process.env.USE_MOCK_DATA = "false"; // Toggle this to "false" to fetch real emails

// Enable CORS for your Chrome extension
app.use(cors({
    origin: "chrome-extension://pdjhkldemalfeibhmgpfffhgipeheekl",
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"],
}));

// Middleware to parse incoming JSON requests
app.use(bodyParser.json());


// Routes
app.use("/api/emails", emailRoutes);


const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
