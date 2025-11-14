"""
Weather query tools using Amap API
"""
import logging
import requests
from typing import Dict, Any, Optional, Tuple
from src.config.settings import settings

logger = logging.getLogger(__name__)


class WeatherService:
    """Weather service using Amap API"""
    
    # Amap API endpoints
    WEATHER_API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
    GEO_API_URL = "https://restapi.amap.com/v3/geocode/geo"
    
    def __init__(self, api_key: str):
        """
        Initialize weather service
        
        Args:
            api_key: Amap API key
        """
        self.api_key = api_key
        logger.info("Weather service initialized successfully")
    
    def get_city_code(self, city_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get city code from city name
        
        Args:
            city_name: City name
            
        Returns:
            Tuple of (city_code, city_name) or (None, None) if not found
        """
        try:
            params = {
                "key": self.api_key,
                "address": city_name,
                "output": "JSON"
            }
            
            response = requests.get(self.GEO_API_URL, params=params)
            data = response.json()
            
            if data["status"] == "1" and data["count"] != "0":
                geocode = data["geocodes"][0]
                return geocode["adcode"], geocode["city"] or geocode["district"]
            else:
                logger.warning(f"City not found: {city_name}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error getting city code: {str(e)}")
            return None, None
    
    def query_weather(self, city: str, extensions: str = "all") -> Dict[str, Any]:
        """
        Query weather information
        
        Args:
            city: City name or code
            extensions: Weather type (base/all)
            
        Returns:
            Weather information dictionary
        """
        result = {
            "status": "error",
            "data": None,
            "message": ""
        }
        
        try:
            # Get city code if needed
            city_code, city_name = city, city
            if not city.isdigit():
                city_code, city_name = self.get_city_code(city)
                
            if not city_code:
                result["message"] = f"City not found: {city}"
                return result
                
            # Query weather API
            params = {
                "key": self.api_key,
                "city": city_code,
                "extensions": extensions,
                "output": "JSON"
            }
            
            response = requests.get(self.WEATHER_API_URL, params=params)
            data = response.json()
            
            if data["status"] == "1":
                result["status"] = "success"
                result["data"] = data
                
                # Format summary
                if extensions == "base":
                    lives = data.get("lives", [])
                    if lives:
                        weather_info = lives[0]
                        result["summary"] = self._format_current_weather(weather_info, city_name)
                else:
                    forecasts = data.get("forecasts", [])
                    if forecasts and forecasts[0].get("casts"):
                        result["summary"] = self._format_forecast_weather(forecasts[0], city_name)
            else:
                result["message"] = f"Weather query failed: {data}"
                
            return result
            
        except Exception as e:
            logger.error(f"Weather query error: {str(e)}")
            result["message"] = f"Weather query error: {str(e)}"
            return result
    
    def _format_current_weather(self, weather: Dict[str, Any], city_name: str) -> str:
        """Format current weather information"""
        return (
            f"Current weather in {city_name}: {weather.get('weather')}, "
            f"temperature {weather.get('temperature')}°C, "
            f"humidity {weather.get('humidity')}%, "
            f"{weather.get('winddirection')} wind level {weather.get('windpower')}. "
            f"Data updated at: {weather.get('reporttime')}"
        )
    
    def _format_forecast_weather(self, forecast: Dict[str, Any], city_name: str) -> str:
        """Format weather forecast information"""
        result = f"Weather forecast for {city_name}:\n"
        
        for cast in forecast.get("casts", []):
            date = cast.get("date")
            day_weather = cast.get("dayweather")
            night_weather = cast.get("nightweather")
            day_temp = cast.get("daytemp")
            night_temp = cast.get("nighttemp")
            day_wind = f"{cast.get('daywind')} wind level {cast.get('daypower')}"
            night_wind = f"{cast.get('nightwind')} wind level {cast.get('nightpower')}"
            
            result += (
                f"{date}: Day: {day_weather} {day_temp}°C {day_wind}, "
                f"Night: {night_weather} {night_temp}°C {night_wind}\n"
            )
            
        return result


class WeatherTools:
    """Weather tools wrapper"""
    
    def __init__(self, api_key: str):
        """
        Initialize weather tools
        
        Args:
            api_key: Amap API key
        """
        self.weather_service = WeatherService(api_key)
        logger.info("Weather tools initialized successfully")
    
    def query_weather(self, city: str) -> str:
        """
        Query weather for a specific city
        
        Args:
            city: City name
            
        Returns:
            Weather information string
        """
        result = self.weather_service.query_weather(city)
        if result["status"] == "success" and "summary" in result:
            return result["summary"]
        else:
            return f"Failed to get weather information for {city}: {result.get('message', 'Unknown error')}"
        
        