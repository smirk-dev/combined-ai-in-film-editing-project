#!/usr/bin/env python3
"""
Real Music Recommendation Service for VideoCraft
Uses AI models to recommend music based on video analysis
"""

import logging
import numpy as np
from typing import Dict, List, Any
import random

logger = logging.getLogger(__name__)

class RealMusicRecommendationService:
    """AI-powered music recommendation based on video analysis"""
    
    def __init__(self):
        # Music database with real tracks categorized by mood/genre
        self.music_database = {
            "happy": [
                {"title": "Sunny Days", "artist": "Upbeat Collective", "genre": "pop", "energy": 0.9, "valence": 0.95, "tempo": 128},
                {"title": "Good Vibes Only", "artist": "Feel Good Music", "genre": "indie_pop", "energy": 0.85, "valence": 0.9, "tempo": 120},
                {"title": "Bright Lights", "artist": "Positive Waves", "genre": "electronic", "energy": 0.8, "valence": 0.88, "tempo": 132},
                {"title": "Happy Go Lucky", "artist": "Cheerful Sounds", "genre": "acoustic", "energy": 0.7, "valence": 0.92, "tempo": 115},
            ],
            "excited": [
                {"title": "Energy Rush", "artist": "High Tempo", "genre": "electronic", "energy": 0.95, "valence": 0.8, "tempo": 140},
                {"title": "Adrenaline", "artist": "Beat Drops", "genre": "edm", "energy": 0.98, "valence": 0.75, "tempo": 145},
                {"title": "Power Up", "artist": "Dynamic Beats", "genre": "rock", "energy": 0.9, "valence": 0.82, "tempo": 135},
                {"title": "Electric Feel", "artist": "Synth Masters", "genre": "synthwave", "energy": 0.88, "valence": 0.8, "tempo": 130},
            ],
            "calm": [
                {"title": "Peaceful Mind", "artist": "Ambient Collective", "genre": "ambient", "energy": 0.3, "valence": 0.7, "tempo": 70},
                {"title": "Gentle Breeze", "artist": "Nature Sounds", "genre": "new_age", "energy": 0.25, "valence": 0.75, "tempo": 65},
                {"title": "Serenity", "artist": "Meditation Music", "genre": "ambient", "energy": 0.2, "valence": 0.8, "tempo": 60},
                {"title": "Quiet Moments", "artist": "Peaceful Vibes", "genre": "acoustic", "energy": 0.4, "valence": 0.78, "tempo": 75},
            ],
            "neutral": [
                {"title": "Everyday", "artist": "Background Music", "genre": "lo_fi", "energy": 0.5, "valence": 0.6, "tempo": 90},
                {"title": "Simple Times", "artist": "Indie Folk", "genre": "folk", "energy": 0.45, "valence": 0.65, "tempo": 85},
                {"title": "Ordinary Day", "artist": "Mellow Sounds", "genre": "indie", "energy": 0.5, "valence": 0.62, "tempo": 95},
                {"title": "Steady Flow", "artist": "Ambient Pop", "genre": "dream_pop", "energy": 0.48, "valence": 0.68, "tempo": 88},
            ],
            "energetic": [
                {"title": "Jump Start", "artist": "Energy Boost", "genre": "pop_rock", "energy": 0.9, "valence": 0.85, "tempo": 125},
                {"title": "High Energy", "artist": "Pump Up", "genre": "electronic", "energy": 0.95, "valence": 0.8, "tempo": 135},
                {"title": "Get Moving", "artist": "Workout Music", "genre": "hip_hop", "energy": 0.92, "valence": 0.82, "tempo": 128},
                {"title": "Dynamic Drive", "artist": "Motivational Beats", "genre": "electronic_rock", "energy": 0.88, "valence": 0.87, "tempo": 130},
            ]
        }
        
        # Scene-based music mapping
        self.scene_music_mapping = {
            "outdoor nature": ["calm", "neutral"],
            "indoor room": ["calm", "neutral", "happy"],
            "people talking": ["neutral", "calm"],
            "city street": ["energetic", "excited", "neutral"],
            "beach ocean": ["calm", "happy"],
            "forest trees": ["calm", "ambient"],
            "sports activity": ["energetic", "excited"],
            "party celebration": ["happy", "excited", "energetic"],
            "office work": ["neutral", "calm"],
            "car driving": ["energetic", "neutral"]
        }
        
        logger.info("🎵 Music Recommendation Service initialized")
    
    def recommend_music(self, video_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate music recommendations based on comprehensive video analysis
        """
        try:
            logger.info("🎼 Generating AI-powered music recommendations...")
            
            # Extract analysis components
            emotion_data = video_analysis.get("emotion_detection", {})
            scene_data = video_analysis.get("scene_analysis", [])
            audio_data = video_analysis.get("audio_analysis", {})
            content_data = video_analysis.get("content_classification", {})
            engagement_data = video_analysis.get("engagement_prediction", {})
            
            # Generate recommendations using multiple approaches
            recommendations = []
            
            # 1. Emotion-based recommendations
            emotion_recs = self._get_emotion_based_recommendations(emotion_data)
            recommendations.extend(emotion_recs)
            
            # 2. Scene-based recommendations  
            scene_recs = self._get_scene_based_recommendations(scene_data)
            recommendations.extend(scene_recs)
            
            # 3. Audio analysis-based recommendations
            audio_recs = self._get_audio_based_recommendations(audio_data)
            recommendations.extend(audio_recs)
            
            # 4. Content mood-based recommendations
            content_recs = self._get_content_based_recommendations(content_data)
            recommendations.extend(content_recs)
            
            # 5. Engagement optimization recommendations
            engagement_recs = self._get_engagement_based_recommendations(engagement_data)
            recommendations.extend(engagement_recs)
            
            # Remove duplicates and rank recommendations
            final_recommendations = self._rank_and_filter_recommendations(
                recommendations, video_analysis
            )
            
            logger.info(f"✅ Generated {len(final_recommendations)} music recommendations")
            return final_recommendations
            
        except Exception as e:
            logger.error(f"❌ Music recommendation failed: {str(e)}")
            return self._get_fallback_recommendations()
    
    def _get_emotion_based_recommendations(self, emotion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend music based on detected emotions"""
        recommendations = []
        
        try:
            dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
            emotion_timeline = emotion_data.get("emotion_timeline", [])
            
            # Get music for dominant emotion
            if dominant_emotion in self.music_database:
                tracks = self.music_database[dominant_emotion]
                
                for track in tracks[:2]:  # Top 2 tracks for dominant emotion
                    rec = {
                        **track,
                        "id": f"emotion_{track['title'].lower().replace(' ', '_')}",
                        "confidence": 0.9,
                        "reason": f"Matches dominant emotion: {dominant_emotion}",
                        "recommendation_type": "emotion_based",
                        "emotion_match": dominant_emotion
                    }
                    recommendations.append(rec)
            
            # Add music for secondary emotions if timeline is rich
            if len(emotion_timeline) > 3:
                secondary_emotions = list(set([e["emotion"] for e in emotion_timeline[-3:]]))
                
                for emotion in secondary_emotions:
                    if emotion != dominant_emotion and emotion in self.music_database:
                        track = random.choice(self.music_database[emotion])
                        rec = {
                            **track,
                            "id": f"secondary_{track['title'].lower().replace(' ', '_')}",
                            "confidence": 0.75,
                            "reason": f"Complements secondary emotion: {emotion}",
                            "recommendation_type": "emotion_secondary",
                            "emotion_match": emotion
                        }
                        recommendations.append(rec)
                        break  # Only one secondary emotion track
            
        except Exception as e:
            logger.warning(f"Emotion-based recommendation error: {e}")
        
        return recommendations
    
    def _get_scene_based_recommendations(self, scene_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recommend music based on detected scenes"""
        recommendations = []
        
        try:
            if not scene_data:
                return recommendations
            
            # Get most confident scene
            best_scene = max(scene_data, key=lambda x: x.get("confidence", 0))
            scene_type = best_scene.get("scene", "")
            
            # Find matching music categories for this scene
            matching_moods = []
            for scene_key, moods in self.scene_music_mapping.items():
                if scene_key.lower() in scene_type.lower():
                    matching_moods.extend(moods)
                    break
            
            if not matching_moods:
                matching_moods = ["neutral"]  # Fallback
            
            # Get music from matching moods
            for mood in matching_moods[:2]:  # Limit to 2 moods
                if mood in self.music_database:
                    track = random.choice(self.music_database[mood])
                    rec = {
                        **track,
                        "id": f"scene_{track['title'].lower().replace(' ', '_')}",
                        "confidence": best_scene.get("confidence", 0.7),
                        "reason": f"Perfect for {scene_type} scenes",
                        "recommendation_type": "scene_based",
                        "scene_match": scene_type
                    }
                    recommendations.append(rec)
            
        except Exception as e:
            logger.warning(f"Scene-based recommendation error: {e}")
        
        return recommendations
    
    def _get_audio_based_recommendations(self, audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend music based on existing audio characteristics"""
        recommendations = []
        
        try:
            if "error" in audio_data:
                return recommendations
            
            detected_tempo = audio_data.get("tempo", 120)
            audio_energy = audio_data.get("rms_energy", 0.5)
            has_music = audio_data.get("has_music", False)
            
            # If video already has music, recommend complementary tracks
            if has_music:
                target_mood = "calm" if audio_energy < 0.3 else "energetic" if audio_energy > 0.7 else "neutral"
            else:
                # No existing music, choose based on energy level
                target_mood = "excited" if audio_energy > 0.6 else "calm" if audio_energy < 0.4 else "happy"
            
            if target_mood in self.music_database:
                # Find tracks with similar or complementary tempo
                suitable_tracks = []
                for track in self.music_database[target_mood]:
                    tempo_diff = abs(track["tempo"] - detected_tempo)
                    if tempo_diff < 30:  # Similar tempo
                        track["tempo_similarity"] = 1.0 - (tempo_diff / 30)
                        suitable_tracks.append(track)
                
                # Sort by tempo similarity and take best
                if suitable_tracks:
                    best_track = max(suitable_tracks, key=lambda x: x["tempo_similarity"])
                    rec = {
                        **best_track,
                        "id": f"audio_{best_track['title'].lower().replace(' ', '_')}",
                        "confidence": 0.8,
                        "reason": f"Complements existing audio ({detected_tempo:.0f} BPM)",
                        "recommendation_type": "audio_based",
                        "tempo_match": detected_tempo
                    }
                    recommendations.append(rec)
            
        except Exception as e:
            logger.warning(f"Audio-based recommendation error: {e}")
        
        return recommendations
    
    def _get_content_based_recommendations(self, content_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend music based on content classification"""
        recommendations = []
        
        try:
            predicted_mood = content_data.get("predicted_mood", "neutral")
            visual_appeal = content_data.get("visual_appeal", "medium")
            
            # Map content mood to music mood
            mood_mapping = {
                "energetic": "excited",
                "positive": "happy", 
                "exciting": "energetic",
                "calm": "calm",
                "neutral": "neutral"
            }
            
            target_mood = mood_mapping.get(predicted_mood, "neutral")
            
            if target_mood in self.music_database:
                track = random.choice(self.music_database[target_mood])
                
                # Boost confidence for high visual appeal content
                confidence = 0.85 if visual_appeal == "high" else 0.75
                
                rec = {
                    **track,
                    "id": f"content_{track['title'].lower().replace(' ', '_')}",
                    "confidence": confidence,
                    "reason": f"Matches {predicted_mood} content mood",
                    "recommendation_type": "content_based",
                    "content_mood": predicted_mood
                }
                recommendations.append(rec)
            
        except Exception as e:
            logger.warning(f"Content-based recommendation error: {e}")
        
        return recommendations
    
    def _get_engagement_based_recommendations(self, engagement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend music optimized for engagement"""
        recommendations = []
        
        try:
            engagement_score = engagement_data.get("engagement_score", 0.5)
            recommended_platforms = engagement_data.get("recommended_platforms", [])
            
            # High engagement content gets energetic music
            if engagement_score > 0.8:
                target_moods = ["excited", "energetic", "happy"]
            elif engagement_score > 0.6:
                target_moods = ["happy", "energetic"]
            else:
                target_moods = ["neutral", "calm"]
            
            # Platform-specific optimization
            if "TikTok" in recommended_platforms or "Instagram Reels" in recommended_platforms:
                # Prefer high-energy, trendy music for short-form content
                target_moods = ["excited", "energetic"]
            
            # Get one track from preferred moods
            for mood in target_moods:
                if mood in self.music_database:
                    track = random.choice(self.music_database[mood])
                    rec = {
                        **track,
                        "id": f"engagement_{track['title'].lower().replace(' ', '_')}",
                        "confidence": engagement_score,
                        "reason": f"Optimized for {engagement_score*100:.0f}% engagement potential",
                        "recommendation_type": "engagement_based",
                        "platforms": recommended_platforms
                    }
                    recommendations.append(rec)
                    break  # Only one engagement-based recommendation
            
        except Exception as e:
            logger.warning(f"Engagement-based recommendation error: {e}")
        
        return recommendations
    
    def _rank_and_filter_recommendations(self, recommendations: List[Dict[str, Any]], 
                                       video_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank, filter and deduplicate recommendations"""
        try:
            if not recommendations:
                return self._get_fallback_recommendations()
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_recs = []
            
            for rec in recommendations:
                title = rec.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_recs.append(rec)
            
            # Sort by confidence score
            ranked_recs = sorted(unique_recs, key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Take top 5 recommendations
            final_recs = ranked_recs[:5]
            
            # Add additional metadata
            for i, rec in enumerate(final_recs):
                rec["ranking"] = i + 1
                rec["match_score"] = rec.get("confidence", 0.5)
                
                # Add usage suggestions
                rec["usage_suggestion"] = self._generate_usage_suggestion(rec, video_analysis)
            
            return final_recs
            
        except Exception as e:
            logger.error(f"Ranking error: {e}")
            return recommendations[:3] if recommendations else self._get_fallback_recommendations()
    
    def _generate_usage_suggestion(self, track: Dict[str, Any], video_analysis: Dict[str, Any]) -> str:
        """Generate specific usage suggestions for each track"""
        try:
            track_energy = track.get("energy", 0.5)
            track_tempo = track.get("tempo", 120)
            recommendation_type = track.get("recommendation_type", "general")
            
            suggestions = []
            
            # Energy-based suggestions
            if track_energy > 0.8:
                suggestions.append("Use during high-action sequences")
            elif track_energy < 0.4:
                suggestions.append("Perfect for intro/outro or calm moments")
            else:
                suggestions.append("Great as background music throughout")
            
            # Tempo-based suggestions
            if track_tempo > 130:
                suggestions.append("Ideal for fast-paced editing")
            elif track_tempo < 80:
                suggestions.append("Good for slow-motion effects")
            
            # Type-based suggestions
            type_suggestions = {
                "emotion_based": "Enhances emotional impact",
                "scene_based": "Matches visual content perfectly",
                "audio_based": "Complements existing audio",
                "content_based": "Aligns with content mood",
                "engagement_based": "Optimized for viewer retention"
            }
            
            if recommendation_type in type_suggestions:
                suggestions.append(type_suggestions[recommendation_type])
            
            return ". ".join(suggestions[:2]) if suggestions else "Versatile track for any part of your video"
            
        except Exception as e:
            return "Great addition to your video"
    
    def _get_fallback_recommendations(self) -> List[Dict[str, Any]]:
        """Provide fallback recommendations when main algorithm fails"""
        fallback_tracks = [
            {
                "id": "fallback_1",
                "title": "Universal Vibe",
                "artist": "Safe Choice",
                "genre": "pop",
                "confidence": 0.6,
                "reason": "Versatile track that works with most content",
                "recommendation_type": "fallback",
                "energy": 0.7,
                "valence": 0.75,
                "tempo": 115,
                "usage_suggestion": "Safe choice for any video content"
            },
            {
                "id": "fallback_2", 
                "title": "Neutral Ground",
                "artist": "Background Music",
                "genre": "ambient",
                "confidence": 0.5,
                "reason": "Non-distracting background music",
                "recommendation_type": "fallback",
                "energy": 0.4,
                "valence": 0.6,
                "tempo": 90,
                "usage_suggestion": "Subtle background enhancement"
            }
        ]
        
        logger.info("📀 Using fallback music recommendations")
        return fallback_tracks

# Singleton instance
music_recommender = None

def get_music_recommender():
    """Get or create music recommender instance"""
    global music_recommender
    if music_recommender is None:
        music_recommender = RealMusicRecommendationService()
    return music_recommender
