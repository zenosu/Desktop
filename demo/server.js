const express = require('express');
const axios = require('axios');
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables from .env file
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const YOUTUBE_API_KEY = process.env.YOUTUBE_API_KEY;
const CHANNEL_ID = 'UC_x36zCEGilGpB1m-V4gmjg'; // Hardcoded channel ID for Indy Dev Dan

// Serve static files
app.use(express.static('public'));

// API Endpoints
app.get('/api/channel', async (req, res) => {
  try {
    const response = await axios.get('https://www.googleapis.com/youtube/v3/channels', {
      params: {
        part: 'snippet,statistics',
        id: CHANNEL_ID,
        key: YOUTUBE_API_KEY
      }
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching channel data:', error.message);
    res.status(500).json({ error: 'Failed to fetch channel data' });
  }
});

app.get('/api/videos', async (req, res) => {
  try {
    // First, get the uploads playlist ID
    const channelResponse = await axios.get('https://www.googleapis.com/youtube/v3/channels', {
      params: {
        part: 'contentDetails',
        id: CHANNEL_ID,
        key: YOUTUBE_API_KEY
      }
    });
    
    const uploadsPlaylistId = channelResponse.data.items[0].contentDetails.relatedPlaylists.uploads;
    let allVideos = [];
    let nextPageToken = null;
    
    // Handle pagination to get all videos
    do {
      const playlistResponse = await axios.get('https://www.googleapis.com/youtube/v3/playlistItems', {
        params: {
          part: 'snippet,contentDetails',
          playlistId: uploadsPlaylistId,
          maxResults: 50,
          pageToken: nextPageToken,
          key: YOUTUBE_API_KEY
        }
      });
      
      allVideos = [...allVideos, ...playlistResponse.data.items];
      nextPageToken = playlistResponse.data.nextPageToken;
    } while (nextPageToken);
    
    // Get video statistics for all videos
    // We need to batch these requests as YouTube API has limits
    const videoIds = allVideos.map(video => video.contentDetails.videoId).join(',');
    
    // Split into chunks of 50 (API limit for video IDs)
    const chunks = [];
    const videoIdArray = allVideos.map(video => video.contentDetails.videoId);
    for (let i = 0; i < videoIdArray.length; i += 50) {
      chunks.push(videoIdArray.slice(i, i + 50));
    }
    
    // Process each chunk
    const videoStatsPromises = chunks.map(chunk => {
      return axios.get('https://www.googleapis.com/youtube/v3/videos', {
        params: {
          part: 'statistics,snippet',
          id: chunk.join(','),
          key: YOUTUBE_API_KEY
        }
      });
    });
    
    const videoStatsResponses = await Promise.all(videoStatsPromises);
    const videoStats = videoStatsResponses.flatMap(response => response.data.items);
    
    // Combine video data with statistics
    const videosWithStats = allVideos.map(video => {
      const stats = videoStats.find(stat => stat.id === video.contentDetails.videoId);
      return {
        id: video.contentDetails.videoId,
        title: video.snippet.title,
        description: video.snippet.description,
        publishedAt: video.snippet.publishedAt,
        thumbnail: video.snippet.thumbnails.medium.url,
        viewCount: stats ? parseInt(stats.statistics.viewCount) : 0,
        likeCount: stats ? parseInt(stats.statistics.likeCount) : 0,
        commentCount: stats ? parseInt(stats.statistics.commentCount) : 0
      };
    });
    
    // Sort by published date
    videosWithStats.sort((a, b) => new Date(a.publishedAt) - new Date(b.publishedAt));
    
    res.json(videosWithStats);
  } catch (error) {
    console.error('Error fetching videos:', error.message);
    res.status(500).json({ error: 'Failed to fetch videos' });
  }
});

// New endpoint for historical view count data
app.get('/api/historical-views', async (req, res) => {
  try {
    // Reuse existing code to fetch videos with stats
    const channelResponse = await axios.get('https://www.googleapis.com/youtube/v3/channels', {
      params: {
        part: 'contentDetails',
        id: CHANNEL_ID,
        key: YOUTUBE_API_KEY
      }
    });
    
    const uploadsPlaylistId = channelResponse.data.items[0].contentDetails.relatedPlaylists.uploads;
    let allVideos = [];
    let nextPageToken = null;
    
    // Handle pagination to get all videos
    do {
      const playlistResponse = await axios.get('https://www.googleapis.com/youtube/v3/playlistItems', {
        params: {
          part: 'snippet,contentDetails',
          playlistId: uploadsPlaylistId,
          maxResults: 50,
          pageToken: nextPageToken,
          key: YOUTUBE_API_KEY
        }
      });
      
      allVideos = [...allVideos, ...playlistResponse.data.items];
      nextPageToken = playlistResponse.data.nextPageToken;
    } while (nextPageToken);
    
    // Get video statistics for all videos
    const chunks = [];
    const videoIdArray = allVideos.map(video => video.contentDetails.videoId);
    for (let i = 0; i < videoIdArray.length; i += 50) {
      chunks.push(videoIdArray.slice(i, i + 50));
    }
    
    const videoStatsPromises = chunks.map(chunk => {
      return axios.get('https://www.googleapis.com/youtube/v3/videos', {
        params: {
          part: 'statistics,snippet',
          id: chunk.join(','),
          key: YOUTUBE_API_KEY
        }
      });
    });
    
    const videoStatsResponses = await Promise.all(videoStatsPromises);
    const videoStats = videoStatsResponses.flatMap(response => response.data.items);
    
    // Combine video data with statistics and sort by published date
    const videosWithStats = allVideos.map(video => {
      const stats = videoStats.find(stat => stat.id === video.contentDetails.videoId);
      return {
        id: video.contentDetails.videoId,
        title: video.snippet.title,
        publishedAt: video.snippet.publishedAt,
        viewCount: stats ? parseInt(stats.statistics.viewCount) : 0,
      };
    }).sort((a, b) => new Date(a.publishedAt) - new Date(b.publishedAt));
    
    // Group videos by time periods (monthly)
    const monthlyData = {};
    let cumulativeViews = 0;
    
    videosWithStats.forEach(video => {
      const date = new Date(video.publishedAt);
      const yearMonth = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      
      if (!monthlyData[yearMonth]) {
        monthlyData[yearMonth] = {
          period: yearMonth,
          videoCount: 0,
          viewsInPeriod: 0,
          cumulativeViews: 0
        };
      }
      
      monthlyData[yearMonth].videoCount++;
      monthlyData[yearMonth].viewsInPeriod += video.viewCount;
      cumulativeViews += video.viewCount;
      monthlyData[yearMonth].cumulativeViews = cumulativeViews;
    });
    
    // Convert to array and ensure chronological order
    const historicalData = Object.values(monthlyData).sort((a, b) => {
      return a.period.localeCompare(b.period);
    });
    
    // Create quarterly data
    const quarterlyData = {};
    
    historicalData.forEach(month => {
      const [year, monthNum] = month.period.split('-');
      const quarter = `${year}-Q${Math.ceil(parseInt(monthNum) / 3)}`;
      
      if (!quarterlyData[quarter]) {
        quarterlyData[quarter] = {
          period: quarter,
          videoCount: 0,
          viewsInPeriod: 0,
          cumulativeViews: 0
        };
      }
      
      quarterlyData[quarter].videoCount += month.videoCount;
      quarterlyData[quarter].viewsInPeriod += month.viewsInPeriod;
      quarterlyData[quarter].cumulativeViews = month.cumulativeViews;
    });
    
    // Convert to array and ensure chronological order
    const quarterlyHistoricalData = Object.values(quarterlyData).sort((a, b) => {
      return a.period.localeCompare(b.period);
    });
    
    // Create yearly data
    const yearlyData = {};
    
    historicalData.forEach(month => {
      const year = month.period.split('-')[0];
      
      if (!yearlyData[year]) {
        yearlyData[year] = {
          period: year,
          videoCount: 0,
          viewsInPeriod: 0,
          cumulativeViews: 0
        };
      }
      
      yearlyData[year].videoCount += month.videoCount;
      yearlyData[year].viewsInPeriod += month.viewsInPeriod;
      yearlyData[year].cumulativeViews = month.cumulativeViews;
    });
    
    // Convert to array and ensure chronological order
    const yearlyHistoricalData = Object.values(yearlyData).sort((a, b) => {
      return a.period.localeCompare(b.period);
    });
    
    res.json({
      monthly: historicalData,
      quarterly: quarterlyHistoricalData,
      yearly: yearlyHistoricalData
    });
  } catch (error) {
    console.error('Error fetching historical view data:', error.message);
    res.status(500).json({ error: 'Failed to fetch historical view data' });
  }
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
}); 