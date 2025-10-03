# ssml_processor.py
"""
SSML Processor for Voice Cloning
Parses SSML into structured segments for prosody, breaks, emphasis, voice change.
"""
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

class SSMLParseException(Exception):
    pass

# Helper to parse time durations

def _parse_time_duration(duration_str: str) -> int:
    duration_str = duration_str.strip().lower()
    if duration_str.endswith('ms'):
        return int(duration_str[:-2])
    if duration_str.endswith('s'):
        return int(float(duration_str[:-1]) * 1000)
    raise SSMLParseException(f"Invalid time format: {duration_str}")

class SSMLProcessor:
    """Parses SSML input into actionable segments."""
    SUPPORTED_TAGS = {"break","emphasis","prosody","voice","say-as","sub","emotion"}

    # Emotion to prosody mapping
    EMOTION_MAPS = {
        'excited': {
            'rate': '+30%',
            'pitch': '+30%',
            'volume': '+20dB',
            'range': '+50%'
        },
        'sad': {
            'rate': '-20%',
            'pitch': '-20%',
            'volume': '-10dB',
            'range': '-30%'
        },
        'whispered': {
            'rate': '-10%',
            'pitch': '-50%',
            'volume': '-20dB',
            'range': '-60%'
        },
        'angry': {
            'rate': '+10%',
            'pitch': '+20%',
            'volume': '+15dB',
            'range': '+40%'
        },
        'intimate': {
            'rate': '-15%',
            'pitch': '-10%',
            'volume': '-15dB',
            'range': '-20%'
        }
    }

    def parse(self, ssml_text: str) -> List[Dict[str,Any]]:
        try:
            root = ET.fromstring(f"<root>{ssml_text}</root>")
        except ET.ParseError as e:
            raise SSMLParseException(f"SSML parse error: {e}")

        segments: List[Dict[str,Any]] = []
        for elem in root.iter():
            if elem.tag == 'root':
                continue
            if elem.tag == 'break':
                time_attr = elem.attrib.get('time') or elem.attrib.get('strength')
                ms = 500
                if time_attr:
                    try:
                        strength_map={'none':0,'x-weak':150,'weak':300,'medium':500,'strong':700,'x-strong':900}
                        ms = strength_map.get(time_attr,_parse_time_duration(time_attr))
                    except: ms=500
                segments.append({'type':'break','duration_ms':ms})
            elif elem.tag=='emphasis':
                lvl=elem.attrib.get('level','moderate')
                text=(elem.text or '').strip()
                segments.append({'type':'emphasis','text':text,'level':lvl})
            elif elem.tag=='prosody':
                txt=(elem.text or '').strip(); params={}
                for k in ('pitch','rate','volume'):
                    v=elem.attrib.get(k)
                    if v: params[k]=v
                segments.append({'type':'prosody','text':txt,'parameters':params})
            elif elem.tag=='voice':
                txt=(elem.text or '').strip(); params={}
                for k in ('name','gender','age','variant'):
                    v=elem.attrib.get(k)
                    if v: params[k]=v
                segments.append({'type':'voice','text':txt,'parameters':params})
            elif elem.tag=='say-as':
                txt=(elem.text or '').strip()
                segments.append({'type':'say-as','text':txt,'interpret_as':elem.attrib.get('interpret-as','characters'),'format':elem.attrib.get('format')})
            elif elem.tag=='sub':
                orig=(elem.text or '').strip(); alias=elem.attrib.get('alias')
                segments.append({'type':'substitution','original':orig,'alias':alias})
            else:
                txt=(elem.text or '').strip()
                if txt: segments.append({'type':'text','text':txt})
            # handle tail text
            if elem.tail and elem.tail.strip():
                segments.append({'type':'text','text':elem.tail.strip()})
            elif elem.tag == 'emotion':
                emotion = elem.attrib.get('name', 'neutral')
                text = (elem.text or '').strip()
                prosody_params = self._emotion_to_prosody(emotion)
                segments.append({
                    'type': 'emotion',
                    'text': text,
                    'emotion': emotion,
                    'prosody': prosody_params
                })
        return segments

    def flatten_segments_to_text(self, segments: List[Dict[str,Any]]) -> str:
        return ' '.join(seg.get('text','') for seg in segments if seg.get('text'))

    def _convert_percentage(self, value: str) -> float:
        """Convert percentage string to float multiplier"""
        if not value.endswith('%'):
            return float(value)
        return float(value[:-1]) / 100.0
        
    def _emotion_to_prosody(self, emotion: str) -> Dict[str, float]:
        """Convert emotion name to prosody parameters"""
        if emotion not in self.EMOTION_MAPS:
            return {}
            
        params = {}
        for key, value in self.EMOTION_MAPS[emotion].items():
            if value.endswith('%'):
                params[key] = self._convert_percentage(value)
            elif value.endswith('dB'):
                params[key] = float(value[:-2])
            else:
                params[key] = float(value)
                
        return params
