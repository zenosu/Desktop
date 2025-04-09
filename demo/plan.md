Create a simple web application that visualizes YouTube channel analytics for Indy Dev Dan (channel ID: UC_x36zCEGilGpB1m-V4gmjg) with the following specifications:



Core Features
1. Display a line chart of video views over time for Indy Dev Dan's YouTube channel
2. Show detailed video information on chart interaction (title, publish date, view count)
3. Basic responsive design



Technical Requirements

Backend

* Create a Node.js/Express server that securely proxies YouTube API requests
* Implement endpoints for channel info, playlist retrieval, and video details
* Handle pagination for all videos from the channel

Frontend
* Create a single HTML file with inline CSS and JavaScript
* Use Chart.js (loaded via CDN) for data visualization
* Implement custom tooltips for video details
* Keep the UI clean and minimal with a dark theme
1. Security/Configuration
* Keep YouTube API key secure on the server side
* Use environment variables for configuration
* Basic error handling for API failures



Implementation Focus
* Hardcode the channel ID (UC_x36zCEGilGpB1m-V4gmjg) for Indy Dev Dan
* Focus on displaying video views chronologically with accurate data
* Create a visually appealing chart that clearly shows the channel's growth
* Ensure the application loads and displays data quickly
