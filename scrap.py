from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

driver = webdriver.Chrome('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/chromedriver.exe')

urls = 'https://www.justwatch.com/kr?providers=atp,wac,wav&release_year_from=2023&release_year_until=2023'

driver.get(urls)

time.sleep(10)
#목록 전부 로드
for i in range(60):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5)

# 셀렉터
links_selector = '//*[@id="base"]/div[3]/div/div[2]/div[1]/div/div/a'

# 제목, 장르, 재생시간, 연령등급, 시놉시스
title_selector = '//*[@id="base"]/div[2]/div/div[2]/div[2]/div[1]/div[1]/div/h1'
genre_selector = '//*[@id="base"]/div[2]/div/div[1]/div/aside/div[1]/div[3]/div[2]/div[2]'
# create_year_selector = '//*[@id="base"]/div[2]/div/div[2]/div[2]/div[1]/div[1]/div/span'
create_year_selector = 'span.text-muted'
running_time_selector = '//*[@id="base"]/div[2]/div/div[1]/div/aside/div[1]/div[3]/div[3]/div[2]'
synopsis_selector = 'p.text-wrap-pre-line.mt-0 span'

# 링크 수집
OTT_links = driver.find_elements(By.XPATH, links_selector)
OTT_links
OTT_video_name = pd.DataFrame(columns=['title', 'genre', 'create_year', 'running_time', 'synopsis'])
new_row = {}

# 각 링크에 대하여 실행
for i in range(len(OTT_links)):

    url = OTT_links[i].get_attribute('href')
    # 새 탭을 연다
    driver.execute_script("window.open('');")

    # 새 탭으로 이동한다
    driver.switch_to.window(driver.window_handles[-1])

    driver.get(url)

    try:
        # 첫 번째 요소가 발견될 때까지 기다립니다.
        element1 = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, synopsis_selector))
        )
        # # 두 번째 요소가 발견될 때까지 기다립니다.
        # element2 = WebDriverWait(driver, 15).until(
        #     EC.presence_of_element_located((By.XPATH, synopsis_selector))
        # )
        
        # # 세 번째 요소
        # element3 = WebDriverWait(driver, 15).until(
        #     EC.presence_of_element_located((By.XPATH, genre_selector))
        # )

        # # 네 번째 요소
        # element4 = WebDriverWait(driver, 15).until(
        #     EC.presence_of_element_located((By.XPATH, create_year_selector))
        # )

        # # 다섯 번째 요소
        # element51 = WebDriverWait(driver, 15).until(
        #     EC.presence_of_element_located((By.XPATH, running_time_selector))
        # )

        # 요소를 찾았을 때 수행할 작업
        # 제목 내용 수집
        time.sleep(0.2)
        title = driver.find_element(By.XPATH, title_selector).text
        genre = driver.find_element(By.XPATH, genre_selector).text
        create_year = driver.find_element(By.CSS_SELECTOR, create_year_selector).text
        running_time = driver.find_element(By.XPATH, running_time_selector).text
        synopsis = driver.find_element(By.CSS_SELECTOR, synopsis_selector).text
    

    except TimeoutException:
        print('skip')

    except NoSuchElementException:
        print('skip')

    # 새 탭을 닫는다
    finally:
        driver.close()

    # 원래 탭으로 되돌아간다
    driver.switch_to.window(driver.window_handles[0])

    new_row = {'title' : title, 'genre' : genre, 'create_year' : create_year,
                'running_time' : running_time, 'synopsis' : synopsis}

    new_row_DF = pd.DataFrame(new_row, index=[0])

    OTT_video_name = pd.concat([OTT_video_name, new_row_DF]).reset_index(drop=True)

    print(f'Title: {title,genre,create_year,running_time,synopsis}')

OTT_video_name

OTT_video_name.to_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/data/others_13.csv', encoding='utf-8')

