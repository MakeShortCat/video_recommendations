from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import pandas as pd
import os
import numpy as np

driver = webdriver.Chrome('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/driver/chromedriver.exe')
# urls = 'https://pedia.watcha.com/ko-KR/search?query=슬램덩크&category=contents'
def watcha_pedia_scrap(cotent_name, driver):
    
    urls = f'https://pedia.watcha.com/ko-KR/search?query={cotent_name}&category=contents'

    driver.get(urls)

    # 영화 링크 위치
    content_selector = 'li.css-8y23cj a'

    try:
        element1 = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, content_selector))
        )
    except TimeoutException:
        driver.execute_script("window.open('');")

    time.sleep(0.2)

    content_links = driver.find_elements(By.CSS_SELECTOR, content_selector)
    
    # 영화/드라마/tv쇼 등등
    content_type = driver.find_element(By.CSS_SELECTOR, content_selector).text.split('\n')
    # 링크 
    url = content_links[0].get_attribute('href')

    #root > div > div.css-1xm32e0 > section > section > div > div > div > ul > div:nth-child(2) > div.css-4tkoly > a > div
    time.sleep(0.2)
    # 새 탭을 연다
    driver.execute_script("window.open('');")

    # 새 탭으로 이동한다
    driver.switch_to.window(driver.window_handles[-1])

    driver.get(url)

    maker_actor_selector = 'ul.e5xrf7a0.css-1br354h-VisualUl-PeopleStackableUl li a'

   
    element2 = WebDriverWait(driver, 3).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, maker_actor_selector))
    )

    content_title_selector = 'div.css-13h49w0-PaneInner.e1svyhwg16 h1'
    content_year_selector = 'div.css-13h49w0-PaneInner.e1svyhwg16 div'

    time.sleep(1)

    # 영화 기본 정보
    watcha_title = driver.find_element(By.CSS_SELECTOR, content_title_selector).text
    year = driver.find_elements(By.CSS_SELECTOR, content_year_selector)[0].text.split(' ・ ')[0]

    watcha_running_time_selector = 'span.css-1t00yeb-OverviewMeta.eokm2782'

    maker_actor = driver.find_elements(By.CSS_SELECTOR, maker_actor_selector)

    maker_actor_list = [i.get_attribute('title') for i in maker_actor]

    watcha_running_time_ele = driver.find_elements(By.CSS_SELECTOR, watcha_running_time_selector)
    nation_age = watcha_running_time_ele[1].text
    rating_selector = 'div.css-og1gu8-ContentRatings.e1svyhwg20'  
    rating = driver.find_element(By.CSS_SELECTOR, rating_selector).text

    time.sleep(0.2)

    # 코멘트 가져오기
    driver.get(url+'/comments')

    time.sleep(1)

    # 아래로 스크롤하여 전부 불러오기
    for i in range(6):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.8)

    #스포일러 방지 버튼 누르기
    spoiler_more = driver.find_elements(By.CSS_SELECTOR, '[aria-label="Accept Spoiler"]')

    time.sleep(0.2)


    for i in range(len(spoiler_more)):
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", spoiler_more[i])
        time.sleep(0.5)
        spoiler_more[i].click()

        
    comment_name_selector = 'div.css-1cvf9dk a'
    comment_name = driver.find_elements(By.CSS_SELECTOR, comment_name_selector)
    comment_selector = f'//*[@id="root"]/div/div[1]/section/section/div/div/div/ul/div/div/a/div/span'  
    comment_rating_list = []

    for i in range(1, len(comment_name)+1): 
        comment_rating_selector = f'//*[@id="root"]/div/div[1]/section/section/div/div/div/ul/div[{i}]/div[1]/div[2]/span'
        if not comment_rating_selector:
            comment_rating = np.nan

            comment_rating_list.append(comment_rating)

        else:
            comment_rating = driver.find_element(By.XPATH, comment_rating_selector).text

            comment_rating_list.append(comment_rating)

    # comment_rating_selector = 'div.css-yqs4xl span'

    comment = driver.find_elements(By.XPATH, comment_selector)

    comment_name_list = [i.get_attribute('title') for i in comment_name]

    comment_list = [i.text for i in comment]

    watcha_content = pd.DataFrame(columns=['title', 'rating', 'nation', 'watcha_create_year','content_type' ,
                                            'age_limit','maker_actor', 'comment_name', 'comment','comment_rating'])

    for i in range(len(comment_name_list)):
        new_row = {'title' : watcha_title, 'rating' : rating, 'nation' : nation_age, 'watcha_create_year' : year, 'content_type' : [content_type],
                     'age_limit' : nation_age, 'maker_actor' : [maker_actor_list],
                    'comment_name' : comment_name_list[i], 'comment' : comment_list[i],'comment_rating' : comment_rating_list[i]}

        new_row_DF = pd.DataFrame(new_row, index=[0])

        watcha_content = pd.concat([watcha_content, new_row_DF]).reset_index(drop=True)

    print(watcha_content)

    driver.close()
    # 원래 탭으로 되돌아간다
    driver.switch_to.window(driver.window_handles[0])


    return watcha_content


# 파일이 저장된 폴더 경로
folder_path = "C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/data"

# 폴더 내 파일 리스트 가져오기
file_list = os.listdir(folder_path)

# 파일 리스트 중 CSV 파일만 선택하여 데이터프레임으로 읽어오기
df_list = []
for file_name in file_list:
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df_list.append(df)

# 모든 데이터프레임을 병합하여 하나의 데이터프레임으로 만들기
merged_df = pd.concat(df_list, axis=0, ignore_index=True)

merged_df.drop('Unnamed: 0', axis=1, inplace=True)

merged_df = merged_df.drop_duplicates(subset=['title']).dropna().reset_index(drop=True)

watcha_content_DF = pd.DataFrame(columns=['title', 'rating', 'nation' , 'watcha_create_year','content_type' ,
                                        'age_limit','maker_actor', 'comment_name', 'comment', 'comment_rating', 'title_num'])

merged_df.loc[merged_df['title']=='사회인']

for i, title in enumerate(merged_df['title'][20392:]):
    try:
        watcha_content_DF = pd.concat([watcha_content_DF, watcha_pedia_scrap(title, driver)]).reset_index(drop=True)
        watcha_content_DF = watcha_content_DF.fillna(i + 20392)
        # watcha_content_DF['title_num'].loc[watcha_content_DF['title_num'] == np.nan] = i

    except TimeoutException:    
        print('TimeoutException')
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        continue

    except NoSuchElementException:
        print('NoSuchElementException')
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        continue

watcha_content_DF.to_csv(f'C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/watcha_data/watcha_last.csv', escapechar='|')

# watcha_content_DF.to_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/watcha_data/watcha_2.csv')


# watcha_content_DF

# merged_df[merged_df['title'] == '프로메테우스']

# merged_df.loc[181]

# watcha_content_DF.tail(50)
