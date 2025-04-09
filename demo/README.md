# YouTube Analytics for Indy Dev Dan

A simple web application that visualizes YouTube channel analytics for Indy Dev Dan's YouTube channel (channel ID: UC_x36zCEGilGpB1m-V4gmjg).

## Features

- Display a line chart of video views over time
- Show detailed video information on chart interaction (title, publish date, view count)
- List top 20 videos by view count 
- Responsive design with clean dark theme

## Technical Details

- **Backend**: Node.js/Express server that securely proxies YouTube API requests
- **Frontend**: Single HTML file with inline CSS and JavaScript
- **Data Visualization**: Chart.js loaded via CDN
- **Security**: YouTube API key stored securely on server side using environment variables

## Setup and Installation

1. Clone this repository
2. Make sure you have Node.js installed
3. Create a `.env` file in the root directory with your YouTube API key:
   ```
   YOUTUBE_API_KEY=your_api_key_here
   PORT=3000
   ```
4. Install dependencies:
   ```
   npm install
   ```
5. Start the server:
   ```
   npm start
   ```
6. Open your browser and navigate to `http://localhost:3000`

## Project Structure

- `server.js` - Express server with API endpoints
- `public/index.html` - Frontend implementation with HTML, CSS, and JavaScript
- `.env` - Environment variables (not included in repository)

## API Endpoints

- `/api/channel` - Returns channel information for Indy Dev Dan
- `/api/videos` - Returns all videos with their statistics, sorted chronologically

## Notes

This application hard-codes the channel ID for Indy Dev Dan (UC_x36zCEGilGpB1m-V4gmjg) as specified in the requirements. 