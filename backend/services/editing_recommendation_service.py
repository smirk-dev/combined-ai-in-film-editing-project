#!/usr/bin/env python3
"""
Advanced Editing Recommendations Service for VideoCraft
Provides intelligent editing suggestions based on AI analysis
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple
import random

logger = logging.getLogger(__name__)

class AdvancedEditingRecommendationService:
    """AI-powered editing recommendations based on comprehensive video analysis"""
    
    def __init__(self):
        self.cut_techniques = {
            "jump_cut": {"confidence_threshold": 0.7, "description": "Quick cut to maintain pace"},
            "match_cut": {"confidence_threshold": 0.8, "description": "Seamless transition between similar elements"},
            "cutaway": {"confidence_threshold": 0.6, "description": "Cut to different angle or subject"},
            "cross_cut": {"confidence_threshold": 0.75, "description": "Alternate between two different scenes"},
            "fade_in": {"confidence_threshold": 0.8, "description": "Smooth entrance effect"},
            "fade_out": {"confidence_threshold": 0.8, "description": "Smooth exit effect"}
        }
        
        self.color_grading_presets = {
            "warm_bright": {
                "temperature": 200,
                "saturation": 1.2,
                "brightness": 0.1,
                "description": "Warm, inviting look perfect for happy content"
            },
            "cool_modern": {
                "temperature": -150,
                "saturation": 0.9,
                "brightness": 0.05,
                "description": "Cool, contemporary look for professional content"
            },
            "vintage_film": {
                "temperature": 100,
                "saturation": 0.8,
                "brightness": -0.05,
                "description": "Nostalgic film look with reduced saturation"
            },
            "high_contrast": {
                "temperature": 0,
                "saturation": 1.3,
                "brightness": 0.15,
                "description": "Punchy, high-contrast look for dynamic content"
            },
            "natural": {
                "temperature": 50,
                "saturation": 1.0,
                "brightness": 0.02,
                "description": "Natural, unprocessed look"
            }
        }
        
        self.transition_effects = {
            "cut": {"complexity": "simple", "description": "Direct cut - clean and fast"},
            "dissolve": {"complexity": "medium", "description": "Smooth blend between clips"},
            "wipe": {"complexity": "medium", "description": "One image replaces another with motion"},
            "slide": {"complexity": "medium", "description": "New clip slides in from edge"},
            "zoom": {"complexity": "advanced", "description": "Zoom transition for dramatic effect"},
            "spin": {"complexity": "advanced", "description": "Rotating transition for energy"}
        }
        
        logger.info("✂️ Advanced Editing Recommendation Service initialized")
    
    def generate_editing_recommendations(self, video_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive editing recommendations based on AI analysis
        """
        try:
            logger.info("🎬 Generating advanced editing recommendations...")
            
            # Extract analysis data
            emotion_data = video_analysis.get("emotion_detection", {})
            scene_data = video_analysis.get("scene_analysis", [])
            audio_data = video_analysis.get("audio_analysis", {})
            motion_data = video_analysis.get("motion_analysis", {})
            content_data = video_analysis.get("content_classification", {})
            engagement_data = video_analysis.get("engagement_prediction", {})
            
            recommendations = {
                "cuts": self._generate_cut_recommendations(emotion_data, scene_data, audio_data),
                "color_grading": self._recommend_color_grading(content_data, emotion_data, scene_data),
                "transitions": self._recommend_transitions(motion_data, engagement_data),
                "effects": self._recommend_effects(content_data, emotion_data, engagement_data),
                "pacing": self._analyze_pacing_recommendations(audio_data, motion_data, emotion_data),
                "text_overlays": self._recommend_text_overlays(emotion_data, engagement_data),
                "audio_adjustments": self._recommend_audio_adjustments(audio_data, scene_data),
                "platform_specific": self._generate_platform_specific_edits(engagement_data)
            }
            
            # Add overall editing strategy
            recommendations["editing_strategy"] = self._create_editing_strategy(
                recommendations, video_analysis
            )
            
            logger.info("✅ Advanced editing recommendations generated successfully")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Editing recommendation generation failed: {str(e)}")
            return self._get_fallback_editing_recommendations()
    
    def _generate_cut_recommendations(self, emotion_data: Dict[str, Any], 
                                    scene_data: List[Dict[str, Any]], 
                                    audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent cut point recommendations"""
        cuts = []
        
        try:
            emotion_timeline = emotion_data.get("emotion_timeline", [])
            
            # Generate cuts based on emotion changes
            for i, emotion in enumerate(emotion_timeline):
                if i > 0:  # Compare with previous emotion
                    prev_emotion = emotion_timeline[i-1]
                    
                    # Significant emotion change = good cut point
                    emotion_change = abs(emotion["intensity"] - prev_emotion["intensity"])
                    
                    if emotion_change > 0.3:  # Significant change
                        cut = {
                            "id": f"emotion_cut_{i}",
                            "timestamp": emotion["timestamp"],
                            "type": "emotion_transition",
                            "reason": f"Emotion change: {prev_emotion['emotion']} → {emotion['emotion']}",
                            "confidence": min(emotion_change + 0.5, 1.0),
                            "cut_technique": "jump_cut" if emotion_change > 0.5 else "dissolve",
                            "description": f"Cut at emotional transition for better flow",
                            "startTime": self._timestamp_to_seconds(prev_emotion["timestamp"]),
                            "endTime": self._timestamp_to_seconds(emotion["timestamp"]),
                            "expected_impact": "Maintains emotional continuity"
                        }
                        cuts.append(cut)
            
            # Generate cuts based on scene changes
            for i, scene in enumerate(scene_data):
                if i > 0:
                    prev_scene = scene_data[i-1]
                    
                    if scene["scene"] != prev_scene["scene"]:
                        cut = {
                            "id": f"scene_cut_{i}",
                            "timestamp": scene["timestamp"],
                            "type": "scene_transition",
                            "reason": f"Scene change: {prev_scene['scene']} → {scene['scene']}",
                            "confidence": min(scene["confidence"] + 0.2, 1.0),
                            "cut_technique": "match_cut",
                            "description": "Cut at natural scene boundary",
                            "startTime": self._timestamp_to_seconds(prev_scene["timestamp"]),
                            "endTime": self._timestamp_to_seconds(scene["timestamp"]),
                            "expected_impact": "Smooth visual transition"
                        }
                        cuts.append(cut)
            
            # Generate cuts based on audio analysis
            if not audio_data.get("error"):
                tempo = audio_data.get("tempo", 120)
                beat_interval = 60 / tempo  # Seconds between beats
                
                # Suggest cuts on beat for rhythmic content
                if tempo > 100:  # Rhythmic content
                    for beat_time in range(2, int(30), int(beat_interval * 4)):  # Every 4 beats
                        cut = {
                            "id": f"beat_cut_{beat_time}",
                            "timestamp": f"00:{beat_time:02d}",
                            "type": "rhythm_cut",
                            "reason": f"Cut on beat ({tempo:.0f} BPM)",
                            "confidence": 0.8,
                            "cut_technique": "jump_cut",
                            "description": "Rhythmic cut synchronized with music",
                            "startTime": beat_time - 1,
                            "endTime": beat_time + 1,
                            "expected_impact": "Creates dynamic rhythm"
                        }
                        cuts.append(cut)
                        
                        if len(cuts) >= 10:  # Limit cuts
                            break
            
            # Sort by confidence and limit results
            cuts = sorted(cuts, key=lambda x: x["confidence"], reverse=True)[:8]
            
            return cuts
            
        except Exception as e:
            logger.warning(f"Cut recommendation error: {e}")
            return self._get_fallback_cuts()
    
    def _recommend_color_grading(self, content_data: Dict[str, Any], 
                               emotion_data: Dict[str, Any],
                               scene_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend color grading based on content analysis"""
        try:
            # Analyze content characteristics
            predicted_mood = content_data.get("predicted_mood", "neutral")
            brightness_score = content_data.get("brightness_score", 0.5)
            saturation_score = content_data.get("saturation_score", 0.5)
            
            # Analyze dominant emotion
            dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
            
            # Analyze scene types
            outdoor_scenes = sum(1 for scene in scene_data if "outdoor" in scene.get("scene", ""))
            indoor_scenes = sum(1 for scene in scene_data if "indoor" in scene.get("scene", ""))
            
            # Choose color grading preset
            if dominant_emotion in ["happy", "excited"] and brightness_score > 0.6:
                preset = "warm_bright"
                confidence = 0.9
            elif predicted_mood == "calm" or outdoor_scenes > indoor_scenes:
                preset = "natural"
                confidence = 0.8
            elif saturation_score > 0.7 and predicted_mood == "energetic":
                preset = "high_contrast"
                confidence = 0.85
            elif brightness_score < 0.4:
                preset = "vintage_film"
                confidence = 0.75
            else:
                preset = "cool_modern"
                confidence = 0.7
            
            grading_rec = {
                "preset": preset,
                "confidence": confidence,
                "reason": f"Matches {dominant_emotion} emotion and {predicted_mood} mood",
                "settings": self.color_grading_presets[preset],
                "description": self.color_grading_presets[preset]["description"],
                "expected_impact": f"Enhances {dominant_emotion} emotional tone"
            }
            
            # Add specific adjustments
            adjustments = []
            
            if brightness_score < 0.4:
                adjustments.append({"type": "brightness", "value": 0.15, "reason": "Brighten dark footage"})
            elif brightness_score > 0.8:
                adjustments.append({"type": "brightness", "value": -0.05, "reason": "Reduce overexposure"})
            
            if saturation_score < 0.4:
                adjustments.append({"type": "saturation", "value": 1.2, "reason": "Boost color richness"})
            elif saturation_score > 0.8:
                adjustments.append({"type": "saturation", "value": 0.9, "reason": "Reduce oversaturation"})
            
            grading_rec["custom_adjustments"] = adjustments
            
            return grading_rec
            
        except Exception as e:
            logger.warning(f"Color grading recommendation error: {e}")
            return {
                "preset": "natural",
                "confidence": 0.6,
                "reason": "Safe default choice",
                "settings": self.color_grading_presets["natural"]
            }
    
    def _recommend_transitions(self, motion_data: Dict[str, Any], 
                             engagement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend transition effects based on motion and engagement analysis"""
        transitions = []
        
        try:
            motion_intensity = motion_data.get("motion_intensity", 0.5)
            engagement_score = engagement_data.get("engagement_score", 0.5)
            
            # High motion/engagement = dynamic transitions
            if motion_intensity > 0.7 and engagement_score > 0.7:
                recommended = ["zoom", "spin", "slide", "wipe"]
            elif motion_intensity > 0.5 or engagement_score > 0.6:
                recommended = ["dissolve", "slide", "wipe"]
            else:
                recommended = ["cut", "dissolve"]
            
            for i, transition_type in enumerate(recommended):
                transition = {
                    "id": f"transition_{i+1}",
                    "type": transition_type,
                    "confidence": 0.8 if i == 0 else 0.6,  # First recommendation has higher confidence
                    "description": self.transition_effects[transition_type]["description"],
                    "complexity": self.transition_effects[transition_type]["complexity"],
                    "reason": self._get_transition_reason(transition_type, motion_intensity, engagement_score),
                    "duration": self._get_transition_duration(transition_type),
                    "usage_tip": self._get_transition_usage_tip(transition_type)
                }
                transitions.append(transition)
            
            return transitions[:4]  # Limit to top 4
            
        except Exception as e:
            logger.warning(f"Transition recommendation error: {e}")
            return [{"type": "cut", "confidence": 0.6, "description": "Simple direct cut"}]
    
    def _recommend_effects(self, content_data: Dict[str, Any], 
                          emotion_data: Dict[str, Any],
                          engagement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend visual effects based on analysis"""
        effects = []
        
        try:
            predicted_mood = content_data.get("predicted_mood", "neutral")
            dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
            engagement_score = engagement_data.get("engagement_score", 0.5)
            
            # Mood-based effects
            if predicted_mood == "energetic" or dominant_emotion == "excited":
                effects.extend([
                    {
                        "id": "speed_ramp",
                        "name": "Speed Ramping",
                        "type": "temporal",
                        "confidence": 0.85,
                        "description": "Vary playback speed for dynamic impact",
                        "settings": {"slow_motion": 0.5, "speed_up": 1.5},
                        "reason": "Enhances exciting moments"
                    },
                    {
                        "id": "zoom_punch",
                        "name": "Zoom Punch",
                        "type": "spatial",
                        "confidence": 0.8,
                        "description": "Quick zoom for emphasis",
                        "settings": {"zoom_factor": 1.2, "duration": 0.3},
                        "reason": "Adds energy to key moments"
                    }
                ])
            
            if dominant_emotion == "happy" or predicted_mood == "positive":
                effects.append({
                    "id": "brightness_boost",
                    "name": "Brightness Boost",
                    "type": "color",
                    "confidence": 0.9,
                    "description": "Subtle brightness increase",
                    "settings": {"brightness": 0.1, "contrast": 1.1},
                    "reason": "Enhances positive emotions"
                })
            
            if engagement_score > 0.8:
                effects.append({
                    "id": "shake_stabilization",
                    "name": "Shake Removal",
                    "type": "stabilization",
                    "confidence": 0.85,
                    "description": "Smooth out camera shake",
                    "settings": {"strength": 0.8},
                    "reason": "Professional look increases engagement"
                })
            
            # Always recommend sharpening for better quality
            effects.append({
                "id": "unsharp_mask",
                "name": "Subtle Sharpening",
                "type": "enhancement",
                "confidence": 0.75,
                "description": "Enhance image sharpness",
                "settings": {"amount": 0.3, "radius": 1.0},
                "reason": "Improves overall video quality"
            })
            
            return effects[:5]  # Limit to top 5
            
        except Exception as e:
            logger.warning(f"Effects recommendation error: {e}")
            return []
    
    def _analyze_pacing_recommendations(self, audio_data: Dict[str, Any],
                                      motion_data: Dict[str, Any],
                                      emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and recommend optimal pacing"""
        try:
            tempo = audio_data.get("tempo", 120)
            motion_intensity = motion_data.get("motion_intensity", 0.5)
            emotion_timeline = emotion_data.get("emotion_timeline", [])
            
            # Calculate emotional variance
            if emotion_timeline:
                intensities = [e["intensity"] for e in emotion_timeline]
                emotional_variance = np.var(intensities) if len(intensities) > 1 else 0
            else:
                emotional_variance = 0
            
            # Determine optimal pacing
            if tempo > 130 and motion_intensity > 0.7:
                pacing = "fast"
                target_cut_frequency = "every 2-3 seconds"
                confidence = 0.9
            elif tempo > 100 or emotional_variance > 0.2:
                pacing = "moderate"
                target_cut_frequency = "every 4-6 seconds"
                confidence = 0.8
            else:
                pacing = "slow"
                target_cut_frequency = "every 6-10 seconds"
                confidence = 0.7
            
            return {
                "recommended_pacing": pacing,
                "target_cut_frequency": target_cut_frequency,
                "confidence": confidence,
                "reasoning": f"Based on {tempo:.0f} BPM audio and {motion_intensity:.1f} motion intensity",
                "emotional_variance": round(emotional_variance, 3),
                "pacing_tips": self._get_pacing_tips(pacing)
            }
            
        except Exception as e:
            logger.warning(f"Pacing analysis error: {e}")
            return {"recommended_pacing": "moderate", "confidence": 0.6}
    
    def _recommend_text_overlays(self, emotion_data: Dict[str, Any],
                               engagement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend text overlay strategies"""
        overlays = []
        
        try:
            recommended_platforms = engagement_data.get("recommended_platforms", [])
            dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
            
            # Platform-specific text recommendations
            if "TikTok" in recommended_platforms or "Instagram Reels" in recommended_platforms:
                overlays.append({
                    "id": "hook_text",
                    "type": "hook",
                    "timing": "0-3 seconds",
                    "content": "Attention-grabbing opening text",
                    "style": "large, bold, contrasting color",
                    "confidence": 0.9,
                    "reason": "Essential for short-form content engagement"
                })
            
            # Emotion-based text
            if dominant_emotion in ["excited", "happy"]:
                overlays.append({
                    "id": "energy_text",
                    "type": "emotional",
                    "timing": "At peak emotion moments",
                    "content": "Exclamation or celebration text",
                    "style": "animated, colorful",
                    "confidence": 0.8,
                    "reason": f"Amplifies {dominant_emotion} emotion"
                })
            
            # Call-to-action text
            overlays.append({
                "id": "cta_text",
                "type": "call_to_action",
                "timing": "Final 3 seconds",
                "content": "Like, share, or follow prompt",
                "style": "subtle but visible",
                "confidence": 0.7,
                "reason": "Increases engagement and follower growth"
            })
            
            return overlays
            
        except Exception as e:
            logger.warning(f"Text overlay recommendation error: {e}")
            return []
    
    def _recommend_audio_adjustments(self, audio_data: Dict[str, Any],
                                   scene_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recommend audio adjustments and enhancements"""
        adjustments = []
        
        try:
            if "error" in audio_data:
                return adjustments
            
            avg_volume = audio_data.get("avg_volume", 0.5)
            peak_volume = audio_data.get("peak_volume", 0.8)
            silence_percentage = audio_data.get("silence_percentage", 0.1)
            
            # Volume adjustments
            if avg_volume < 0.3:
                adjustments.append({
                    "id": "volume_boost",
                    "type": "volume",
                    "adjustment": "Increase by 6-10 dB",
                    "confidence": 0.9,
                    "reason": "Audio levels too low",
                    "priority": "high"
                })
            elif peak_volume > 0.95:
                adjustments.append({
                    "id": "volume_normalize",
                    "type": "volume",
                    "adjustment": "Normalize and compress",
                    "confidence": 0.85,
                    "reason": "Prevent audio clipping",
                    "priority": "high"
                })
            
            # Noise reduction
            adjustments.append({
                "id": "noise_reduction",
                "type": "enhancement",
                "adjustment": "Apply gentle noise reduction",
                "confidence": 0.75,
                "reason": "Improve audio quality",
                "priority": "medium"
            })
            
            # Scene-based audio
            outdoor_scenes = sum(1 for scene in scene_data if "outdoor" in scene.get("scene", ""))
            if outdoor_scenes > len(scene_data) / 2:
                adjustments.append({
                    "id": "wind_reduction",
                    "type": "environmental",
                    "adjustment": "High-pass filter for wind noise",
                    "confidence": 0.8,
                    "reason": "Many outdoor scenes detected",
                    "priority": "medium"
                })
            
            return adjustments
            
        except Exception as e:
            logger.warning(f"Audio adjustment recommendation error: {e}")
            return []
    
    def _generate_platform_specific_edits(self, engagement_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate platform-specific editing recommendations"""
        platform_edits = {}
        
        try:
            recommended_platforms = engagement_data.get("recommended_platforms", ["YouTube"])
            
            for platform in recommended_platforms:
                if platform == "TikTok":
                    platform_edits[platform] = [
                        {
                            "edit": "Vertical aspect ratio (9:16)",
                            "priority": "critical",
                            "reason": "TikTok requires vertical format"
                        },
                        {
                            "edit": "Hook in first 3 seconds",
                            "priority": "critical",
                            "reason": "Essential for TikTok algorithm"
                        },
                        {
                            "edit": "Fast-paced editing",
                            "priority": "high",
                            "reason": "Matches platform expectations"
                        }
                    ]
                
                elif platform == "Instagram Reels":
                    platform_edits[platform] = [
                        {
                            "edit": "9:16 vertical format",
                            "priority": "critical",
                            "reason": "Instagram Reels format requirement"
                        },
                        {
                            "edit": "Trending audio",
                            "priority": "high",
                            "reason": "Boosts discoverability"
                        },
                        {
                            "edit": "Text overlays",
                            "priority": "medium",
                            "reason": "Increases engagement"
                        }
                    ]
                
                elif platform == "YouTube":
                    platform_edits[platform] = [
                        {
                            "edit": "16:9 landscape format",
                            "priority": "critical",
                            "reason": "Standard YouTube format"
                        },
                        {
                            "edit": "Strong thumbnail moment",
                            "priority": "high",
                            "reason": "Critical for click-through rate"
                        },
                        {
                            "edit": "Clear audio",
                            "priority": "high",
                            "reason": "YouTube values audio quality"
                        }
                    ]
            
            return platform_edits
            
        except Exception as e:
            logger.warning(f"Platform-specific edit error: {e}")
            return {}
    
    def _create_editing_strategy(self, recommendations: Dict[str, Any], 
                               video_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create overall editing strategy based on all recommendations"""
        try:
            # Analyze the type of content
            engagement_score = video_analysis.get("engagement_prediction", {}).get("engagement_score", 0.5)
            dominant_emotion = video_analysis.get("emotion_detection", {}).get("dominant_emotion", "neutral")
            
            if engagement_score > 0.8:
                strategy_type = "high_engagement"
                approach = "Dynamic, fast-paced editing with strong hooks"
            elif dominant_emotion in ["happy", "excited"]:
                strategy_type = "emotional_positive"
                approach = "Upbeat editing that amplifies positive emotions"
            elif dominant_emotion == "calm":
                strategy_type = "contemplative"
                approach = "Smooth, flowing editing with longer cuts"
            else:
                strategy_type = "balanced"
                approach = "Versatile editing suitable for wide audience"
            
            return {
                "strategy_type": strategy_type,
                "approach": approach,
                "key_focus": self._get_strategy_focus(recommendations),
                "editing_priority": self._get_editing_priorities(recommendations),
                "expected_outcome": self._predict_editing_outcome(strategy_type, engagement_score)
            }
            
        except Exception as e:
            logger.warning(f"Strategy creation error: {e}")
            return {"strategy_type": "balanced", "approach": "Standard editing approach"}
    
    # Helper methods
    def _timestamp_to_seconds(self, timestamp: str) -> int:
        """Convert MM:SS timestamp to seconds"""
        try:
            parts = timestamp.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0
    
    def _get_transition_reason(self, transition_type: str, motion_intensity: float, engagement_score: float) -> str:
        """Get reason for transition recommendation"""
        if transition_type in ["zoom", "spin"] and motion_intensity > 0.7:
            return "Dynamic transition matches high motion content"
        elif transition_type == "dissolve":
            return "Smooth transition for professional look"
        elif transition_type == "cut":
            return "Clean, fast transition maintains pace"
        else:
            return f"Good match for {engagement_score:.1f} engagement level"
    
    def _get_transition_duration(self, transition_type: str) -> float:
        """Get recommended duration for transition"""
        durations = {
            "cut": 0.0,
            "dissolve": 0.5,
            "wipe": 0.3,
            "slide": 0.4,
            "zoom": 0.6,
            "spin": 0.8
        }
        return durations.get(transition_type, 0.3)
    
    def _get_transition_usage_tip(self, transition_type: str) -> str:
        """Get usage tip for transition"""
        tips = {
            "cut": "Use between related shots",
            "dissolve": "Great for time passage or mood changes",
            "wipe": "Effective for location changes",
            "slide": "Works well with horizontal motion",
            "zoom": "Use sparingly for dramatic moments",
            "spin": "Perfect for reveal moments"
        }
        return tips.get(transition_type, "Use creatively")
    
    def _get_pacing_tips(self, pacing: str) -> List[str]:
        """Get pacing-specific tips"""
        tips = {
            "fast": [
                "Keep cuts under 3 seconds",
                "Match cuts to musical beats",
                "Use quick transitions",
                "Maintain high energy throughout"
            ],
            "moderate": [
                "Mix short and medium cuts",
                "Allow moments to breathe",
                "Use varied transition speeds",
                "Balance energy and relaxation"
            ],
            "slow": [
                "Hold shots longer for impact",
                "Use smooth, slow transitions",
                "Let emotions develop naturally",
                "Focus on visual composition"
            ]
        }
        return tips.get(pacing, ["Edit at natural pace"])
    
    def _get_strategy_focus(self, recommendations: Dict[str, Any]) -> List[str]:
        """Determine key focus areas for editing strategy"""
        focus_areas = []
        
        if len(recommendations.get("cuts", [])) > 5:
            focus_areas.append("Dynamic cutting")
        
        if recommendations.get("color_grading", {}).get("confidence", 0) > 0.8:
            focus_areas.append("Color enhancement")
        
        if len(recommendations.get("effects", [])) > 3:
            focus_areas.append("Visual effects")
        
        if recommendations.get("audio_adjustments"):
            focus_areas.append("Audio optimization")
        
        return focus_areas[:3]  # Top 3 focus areas
    
    def _get_editing_priorities(self, recommendations: Dict[str, Any]) -> List[str]:
        """Get editing priorities in order"""
        priorities = []
        
        # Always start with basic adjustments
        priorities.append("Color correction and grading")
        priorities.append("Audio level optimization")
        
        # Add based on confidence levels
        if recommendations.get("pacing", {}).get("confidence", 0) > 0.8:
            priorities.append("Pacing and rhythm")
        
        if len(recommendations.get("cuts", [])) > 3:
            priorities.append("Strategic cutting")
        
        priorities.append("Final polish and effects")
        
        return priorities[:5]
    
    def _predict_editing_outcome(self, strategy_type: str, engagement_score: float) -> str:
        """Predict the outcome of following the editing strategy"""
        outcomes = {
            "high_engagement": f"High viewer retention and {engagement_score*100:.0f}% engagement rate",
            "emotional_positive": "Strong emotional connection and positive viewer response",
            "contemplative": "Thoughtful, immersive viewing experience",
            "balanced": f"Broad appeal with {engagement_score*100:.0f}% predicted engagement"
        }
        return outcomes.get(strategy_type, "Professional, polished final video")
    
    def _get_fallback_cuts(self) -> List[Dict[str, Any]]:
        """Fallback cuts if main algorithm fails"""
        return [
            {
                "id": "fallback_cut_1",
                "timestamp": "00:05",
                "type": "standard",
                "reason": "Natural cut point",
                "confidence": 0.6,
                "cut_technique": "cut",
                "description": "Standard editing cut",
                "startTime": 4,
                "endTime": 6
            }
        ]
    
    def _get_fallback_editing_recommendations(self) -> Dict[str, Any]:
        """Fallback recommendations if main algorithm fails"""
        return {
            "cuts": self._get_fallback_cuts(),
            "color_grading": {
                "preset": "natural",
                "confidence": 0.6,
                "reason": "Safe default",
                "settings": self.color_grading_presets["natural"]
            },
            "transitions": [{"type": "cut", "confidence": 0.6}],
            "effects": [],
            "pacing": {"recommended_pacing": "moderate", "confidence": 0.6},
            "editing_strategy": {"strategy_type": "balanced", "approach": "Standard editing"}
        }

# Singleton instance
editing_recommender = None

def get_editing_recommender():
    """Get or create editing recommender instance"""
    global editing_recommender
    if editing_recommender is None:
        editing_recommender = AdvancedEditingRecommendationService()
    return editing_recommender
