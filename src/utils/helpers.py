BEER_ADVOCATE_FOLDER = 'BeerAdvocate/' #BA
RATE_BEER_FOLDER = 'RateBeer/' #RB
MATCHED_BEER_FOLDER = 'matched_beer_data/' #MB

BA_BEERS_DATASET = BEER_ADVOCATE_FOLDER + "beers.csv"
BA_BREWERIES_DATASET = BEER_ADVOCATE_FOLDER + "breweries.csv"
BA_USERS_DATASET = BEER_ADVOCATE_FOLDER + "users.csv"
BA_RATINGS_DATASET = BEER_ADVOCATE_FOLDER + 'ratings.txt/' + "ratings.txt"
BA_REVIEWS_DATASET = BEER_ADVOCATE_FOLDER + 'reviews.txt/' + "reviews.txt"

RB_BEERS_DATASET = RATE_BEER_FOLDER + "beers.csv"
RB_BREWERIES_DATASET = RATE_BEER_FOLDER + "breweries.csv"
RB_USERS_DATASET = RATE_BEER_FOLDER + "users.csv"
RB_RATINGS_DATASET = RATE_BEER_FOLDER + 'ratings.txt/' + "ratings.txt"
RB_REVIEWS_DATASET = RATE_BEER_FOLDER + 'reviews.txt/' + "ratings.txt"

MB_BEERS_DATASET = MATCHED_BEER_FOLDER + "beers.csv"
MB_BREWERIES_DATASET = MATCHED_BEER_FOLDER + "breweries.csv"
MB_USERS_DATASET = MATCHED_BEER_FOLDER + "users.csv"
MB_USERS_APPROX_DATASET = MATCHED_BEER_FOLDER + "users_approx.csv"
MB_RATINGS_DATASET = MATCHED_BEER_FOLDER + "ratings.csv"

country_continent_map = {
    'Kyrgyzstan': 'Asia', 'Gabon': 'Africa', 'Northern Ireland': 'Europe',
    'Wales': 'Europe', 'Scotland': 'Europe', 'England': 'Europe',
    'Singapore': 'Asia', 'China': 'Asia', 'Chad': 'Africa', 
    'Saint Lucia': 'North America', 'Cameroon': 'Africa',
    'Burkina Faso': 'Africa', 'Zambia': 'Africa', 'Romania': 'Europe',
    'Nigeria': 'Africa', 'South Korea': 'Asia', 'Georgia': 'Asia',
    'Hong Kong': 'Asia', 'Guinea': 'Africa', 'Montenegro': 'Europe',
    'Benin': 'Africa', 'Mexico': 'North America', 'Fiji Islands': 'Oceania',
    'Guam': 'Oceania', 'Laos': 'Asia', 'Senegal': 'Africa',
    'Honduras': 'North America', 'Morocco': 'Africa', 'Indonesia': 'Asia',
    'Monaco': 'Europe', 'Ukraine': 'Europe', 'Canada': 'North America',
    'Jordan': 'Asia', 'Portugal': 'Europe', 'Guernsey': 'Europe',
    'India': 'Asia', 'Puerto Rico': 'North America', 'Japan': 'Asia',
    'Iran': 'Asia', 'Hungary': 'Europe', 'Bulgaria': 'Europe',
    'Guinea-Bissau': 'Africa', 'Liberia': 'Africa', 'Togo': 'Africa',
    'Niger': 'Africa', 'Croatia': 'Europe', 'Lithuania': 'Europe',
    'Cyprus': 'Asia', 'Italy': 'Europe', 'Andorra': 'Europe',
    'Botswana': 'Africa', 'Turks and Caicos Islands': 'North America',
    'Papua New Guinea': 'Oceania', 'Mongolia': 'Asia', 'Ethiopia': 'Africa',
    'Denmark': 'Europe', 'French Polynesia': 'Oceania', 'Greece': 'Europe',
    'Sri Lanka': 'Asia', 'Syria': 'Asia', 'Germany': 'Europe', 'Jersey': 'Europe',
    'Armenia': 'Asia', 'Mozambique': 'Africa', 'Palestine': 'Asia',
    'Bangladesh': 'Asia', 'Turkmenistan': 'Asia', 'Reunion': 'Africa',
    'Eritrea': 'Africa', 'Switzerland': 'Europe', 'Malta': 'Europe',
    'Israel': 'Asia', 'El Salvador': 'North America', 'French Guiana': 'South America',
    'Tonga': 'Oceania', 'Zimbabwe': 'Africa', 'Samoa': 'Oceania', 'Barbados': 'North America',
    'Chile': 'South America', 'Cambodia': 'Asia', 'Cook Islands': 'Oceania',
    'Trinidad & Tobago': 'North America', 'Bhutan': 'Asia', 'Uzbekistan': 'Asia',
    'Egypt': 'Africa', 'Uruguay': 'South America', 'Dominican Republic': 'North America',
    'Equatorial Guinea': 'Africa', 'Russia': 'Europe', 'Tajikistan': 'Asia',
    'Vietnam': 'Asia', 'Palau': 'Oceania', 'Namibia': 'Africa',
    'Cayman Islands': 'North America', 'Sao Tome and Principe': 'Africa', 'Australia': 'Oceania',
    'Martinique': 'North America', 'Virgin Islands (British)': 'North America',
    'Ecuador': 'South America', 'Vanuatu': 'Oceania', 'Congo': 'Africa',
    'Uganda': 'Africa', 'Mauritius': 'Africa', 'Azerbaijan': 'Asia',
    'Argentina': 'South America', 'Tunisia': 'Africa', 'Belize': 'North America',
    'Luxembourg': 'Europe', 'Madagascar': 'Africa', 'Aruba': 'North America',
    'Spain': 'Europe', 'Swaziland': 'Africa', 'South Sudan': 'Africa',
    'Belarus': 'Europe', 'Ivory Coast': 'Africa', 'Austria': 'Europe',
    'Bolivia': 'South America', 'Central African Republic': 'Africa',
    'Mali': 'Africa', 'Suriname': 'South America', 'Solomon Islands': 'Oceania',
    'Rwanda': 'Africa', 'Brazil': 'South America', 'Gibraltar': 'Europe',
    'Taiwan': 'Asia', 'Turkey': 'Asia', 'Greenland': 'North America',
    'Moldova': 'Europe', 'Haiti': 'North America', 'Guadeloupe': 'North America',
    'South Africa': 'Africa', 'Lesotho': 'Africa', 'Czech Republic': 'Europe',
    'Micronesia': 'Oceania', 'Paraguay': 'South America', 'Iraq': 'Asia',
    'Faroe Islands': 'Europe', 'Panama': 'North America', 'Netherlands': 'Europe',
    'Peru': 'South America', 'New Zealand': 'Oceania', 'Ghana': 'Africa',
    'Slovenia': 'Europe', 'Serbia': 'Europe', 'Macedonia': 'Europe',
    'Latvia': 'Europe', 'Guatemala': 'North America', 'Cuba': 'North America',
    'Venezuela': 'South America', 'Angola': 'Africa', 'Finland': 'Europe',
    'Nicaragua': 'North America', 'Sweden': 'Europe', 'Seychelles': 'Africa',
    'Poland': 'Europe', 'Cape Verde Islands': 'Africa', 'Libya': 'Africa',
    'Isle of Man': 'Europe', 'Ireland': 'Europe', 'Myanmar': 'Asia',
    'Algeria': 'Africa', 'Kazakhstan': 'Asia', 'Norway': 'Europe',
    'United States': 'North America', 'Costa Rica': 'North America',
    'North Korea': 'Asia', 'Bosnia and Herzegovina': 'Europe', 'Jamaica': 'North America',
    'Lebanon': 'Asia', 'Dominica': 'North America', 'Virgin Islands (U.S.)': 'North America',
    'Colombia': 'South America', 'Iceland': 'Europe', 'Macau': 'Asia',
    'Grenada': 'North America', 'Malaysia': 'Asia', 'Belgium': 'Europe',
    'Saint Vincent and The Grenadines': 'North America', 'Bahamas': 'North America',
    'Philippines': 'Asia', 'Curaçao': 'North America', 'San Marino': 'Europe',
    'France': 'Europe', 'Bermuda': 'North America', 'Mayotte': 'Africa',
    'Antigua & Barbuda': 'North America', 'Estonia': 'Europe', 'Gambia': 'Africa',
    'Pakistan': 'Asia', 'New Caledonia': 'Oceania', 'Slovak Republic': 'Europe',
    'Liechtenstein': 'Europe', 'Tanzania': 'Africa', 'Malawi': 'Africa',
    'Nepal': 'Asia', 'United Arab Emirates': 'Asia', 'Kenya': 'Africa',
    'Thailand': 'Asia', 'Albania': 'Europe', 'Canada, Ontario': 'North America',
    'United Kingdom, England': 'Europe', 'Canada, Manitoba': 'North America',
    'Canada, Nova Scotia': 'North America', 'Canada, Quebec': 'North America',
    'Canada, Newfoundland and Labrador': 'North America', 'Canada, Alberta': 'North America',
    'Canada, British Columbia': 'North America', 'Canada, Saskatchewan': 'North America',
    'UNKNOWN': 'Unknown', 'Canada, New Brunswick': 'North America',
    'United Kingdom, Wales': 'Europe', 'United Kingdom, Scotland': 'Europe'
}

style_classification = {
    "Lager": ["lager"],
    "Ale": ["ale"],
    "IPA": ["ipa"],
    "Stout/Porter": ["stout", "porter"],
    "Wheat/Sour": ["wheat", "sour"],
    "Bitter": ["bitter"],
    "Saké": ["saké"],
}

def name_to_country(name: str) -> str:
    '''
    Determines the country associated with a given name string
    based on specific formatting rules.
    :param name: str, a string representing a geographical or generic name.
    :return: str, the formatted country name or the original input.
    '''
    if len(name) >= 13:
        if name.split('<')[0] in ['United States', 'Utah', 'New York', 'Illinois']:
            return 'United States'
        if name.split(',')[0] in ['United States']:
            return 'United States'
    return name

def style_to_type(style: str) -> str:
    '''
    Classifies a given beer style into a broader type 
    based on predefined keywords.
    :param style: str, a string representing a beer style.
    :return: str, the broader beer type or "Other" if no match is found.
    '''
    for k, v in style_classification.items():
        for keyword in v:
            if keyword in style.lower():
                return k
    return "Other"