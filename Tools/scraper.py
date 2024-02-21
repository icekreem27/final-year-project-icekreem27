import os, requests, json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def transfer_cookies_to_requests(session, driver):
    cookies = driver.get_cookies()
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

def login():
    # Navigate to the login page
    # (Data Mining module page)
    driver.get('https://minerva.leeds.ac.uk/ultra/courses/_541874_1/outline')
    # Initialize WebDriverWait instance
    wait = WebDriverWait(driver, 10)

    # Wait for the email field to be visible and input the email
    wait.until(EC.visibility_of_element_located((By.ID, 'i0116'))).send_keys(email)
    # Click the next button after entering the email
    wait.until(EC.element_to_be_clickable((By.ID, 'idSIButton9'))).click()

    # Wait for the password field to be visible and input the password
    wait.until(EC.visibility_of_element_located((By.ID, 'i0118'))).send_keys(password)
    # Click the sign in button after entering the password
    wait.until(EC.element_to_be_clickable((By.ID, 'idSIButton9'))).click()

    # DUO
    # Wait for DUO to load and press the passcode button
    wait.until(EC.element_to_be_clickable((By.ID, 'passcode'))).click()
    # Take user inputted password
    duoPassword = input("Enter DUO Password: ")
    # Input password into textbox
    wait.until(EC.visibility_of_element_located((By.NAME, 'passcode'))).send_keys(duoPassword)
    # Click the sign in button after
    wait.until(EC.element_to_be_clickable((By.ID, 'passcode'))).click()

def getUnits():
    # Initialize WebDriverWait instance
    wait = WebDriverWait(driver, 10)
    # Click the Learning Resources button
    wait.until(EC.element_to_be_clickable((By.ID, 'learning-module-title-_9911771_1'))).click()

    # Wait for the units to load and then collect all hrefs
    wait = WebDriverWait(driver, 10)
    
    # Wait for the container of the unit links to be visible
    container = wait.until(EC.visibility_of_element_located(
        (By.ID, 'learning-module-contents-_9911771_1')
    ))

    # Find all <a> tags within the container
    unit_links = container.find_elements(By.TAG_NAME, 'a')

    # Extract hrefs and titles from units
    units_info = [{'url': link.get_attribute('href'), 'title': link.text} for link in unit_links]

    return units_info

def find_files(url, download_directory):
    session = requests.Session()
    transfer_cookies_to_requests(session, driver)

    driver.get(url)
    wait = WebDriverWait(driver, 10)
    
    # Check if containers exist
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-bbtype='attachment']")))
        file_containers = driver.find_elements(By.CSS_SELECTOR, "div[data-bbtype='attachment']")
        
        # For all containers
        for container in file_containers:
            data_bbfile = container.get_attribute("data-bbfile")
            data_href = container.get_attribute("href")
            if data_bbfile:
                data_bbfile_json = json.loads(data_bbfile)
                # Check if files are "lectures" or "transcripts"
                if 'lecture' in data_bbfile_json['linkName'].lower() or 'transcript' in data_bbfile_json['linkName'].lower():
                    download_url = data_href
                    filename = data_bbfile_json['linkName']
                    print(f"Found file: {filename}")
                    download_file(download_url, download_directory, filename, session)
    except TimeoutException:
        print(f"No files found on {url}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def download_file(download_url, download_directory, filename, session):
    path = os.path.join(download_directory, filename)
    response = session.get(download_url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} to {download_directory}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

# Main run
# Access credentials stored in environment variables
email = os.environ.get('MY_EMAIL')
password = os.environ.get('MY_PASSWORD')

# Set up Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Set up a directory to save files
download_directory = 'Data Mining files'
os.makedirs(download_directory, exist_ok=True)

# Call login method
login()

# Fetch Unit Urls
units_info = getUnits()

# Download files for each unit
for unit in units_info:
    print(f"Processing Unit: {unit['title']}")
    find_files(unit['url'], download_directory)

# Always remember to close the WebDriver session when you're done
driver.quit()