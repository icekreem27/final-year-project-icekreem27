from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# Access credentials stored in environment variables
email = os.environ.get('MY_EMAIL')
password = os.environ.get('MY_PASSWORD')

# Set up Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Navigate to the login page
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


# Always remember to close the WebDriver session when you're done
driver.quit()
