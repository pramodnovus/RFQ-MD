import pandas as pd
import re
import pycountry
from rapidfuzz import fuzz, process
from geopy.geocoders import Nominatim
import time

# Setup geolocator
geolocator = Nominatim(user_agent="geoapi")

# Step 1: Load your data
#df = pd.read_csv("output.csv", encoding="ISO-8859-1")  # Use 'utf-8' if no encoding issues
def clean_fields(df):

    def clean_target_group_short(value):
        if pd.isna(value):
            return ""
        text = str(value).strip().lower()
        text = re.split(r"\b(of|from|with|in|for)\b", text)[0].strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.title()

    df["target_group_cleaned"] = df["target_group"].apply(clean_target_group_short)

    bucket_keywords = {
        "healthcare": ["patient", "doctor", "hcp", "physician", "medical", "hospital", "oncologist", "nurse", "colitis", "healthcare"],
        "it": ["software", "technology", "platform", "ai", "cloud", "data", "developer", "tech", "digital", "cyber"],
        "b2b": ["executive", "manager", "director", "supplier", "manufacturer", "repair", "retailer", "distributor", "garage", "tradesman"],
        "finance": ["finance", "loan", "investor", "bank", "acquirer", "fintech"],
        "consumer": ["citizen", "voter", "consumer", "individual", "adult", "user", "public", "people"]
    }

    def assign_bucket(text):
        text = text.lower()
        for bucket, keywords in bucket_keywords.items():
            if any(keyword in text for keyword in keywords):
                return bucket
        return "other"

    df["target_group_bucket"] = df["target_group_cleaned"].apply(assign_bucket)
    df.drop('target_group', axis=1, inplace=True)

    def extract_minutes(loi):
        if pd.isna(loi):
            return None
        text = str(loi).lower().strip()
        text = text.replace("-", " ").replace("–", " ").replace("to", " ")

        match = re.search(r'(\d+)\s*(hour|hr)[s]?\s*(\d+)?\s*(minute|min|mins)?', text)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(3)) if match.group(3) else 0
            return hours * 60 + minutes

        match = re.search(r'(\d+)\s*(minute|min|mins)', text)
        if match:
            return int(match.group(1))

        match = re.search(r'(\d+)\s+(\d+)\s*(minute|min|mins)', text)
        if match:
            return int(match.group(1))

        match = re.fullmatch(r'\d{1,3}', text)
        if match:
            return int(text)

        return None

    df["loi_minutes"] = df["loi"].apply(extract_minutes).astype('Int64')
    df.drop('loi', axis=1, inplace=True)

    non_location_keywords = [
        'total sample', 'language', 'languages', 'n=', 'sample', 'approx',
        'respondents', 'split', 'quota', 'national rep', 'cawi', 'cati', 'web', 'method'
    ]

    def clean_location(value):
        if pd.isna(value):
            return ""
        value = str(value).lower()
        for keyword in non_location_keywords:
            value = re.sub(rf'\b{re.escape(keyword)}\b', '', value)

        value = re.sub(r'\(.*?\)|\[.*?\]', '', value)
        value = re.sub(r'\d+%?', '', value)
        value = re.sub(r'[^a-zA-Z, ]', '', value)
        value = re.sub(r'\s+', ' ', value).strip()
        value = value.strip(", ")
        locations = [loc.strip().title() for loc in value.split(',') if loc.strip()]
        return ', '.join(sorted(set(locations), key=locations.index))

    all_countries = [country.name for country in pycountry.countries]

    def match_to_country(name):
        result = process.extractOne(name, all_countries, scorer=fuzz.ratio)
        if result:
            match, score, _ = result
            return match if score >= 80 else None
        return None

    def get_country(location):
        try:
            time.sleep(1)
            loc = geolocator.geocode(location, language='en')
            if loc:
                location_detail = geolocator.reverse((loc.latitude, loc.longitude), language='en')
                return location_detail.raw['address'].get('country', location)
        except:
            return location

    def validate_locations(value):
        if not value:
            return ""
        cleaned = []
        for loc in value.split(','):
            loc = loc.strip()
            if not loc:
                continue
            country = match_to_country(loc)
            if not country:
                country = get_country(loc)
            if country:
                cleaned.append(country)
        return ', '.join(sorted(set(cleaned), key=cleaned.index))

    df["location_cleaned"] = df["location"].apply(clean_location).apply(validate_locations)
    df.drop('location', axis=1, inplace=True)

    df["id"] = range(1, len(df) + 1)
    df.drop('project_type', axis=1, inplace=True)

    return df

