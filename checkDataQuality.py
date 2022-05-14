import webbrowser
from zmq import VMCI_BUFFER_SIZE
import os
from loadData import getOutputDataPaths
import subprocess
from time import sleep
import win32gui as wg
from win32gui import GetForegroundWindow
from win32api import GetKeyState

def keyIsDown(key):
    keystate = GetKeyState( key )
    if (keystate != 0) and (keystate != 1):
        return True
    else:
        return False
# Key codes: https://pastebin.com/Bj0JmmD3
def getRating():
    while True:
        if keyIsDown(71): return 'g'
        elif keyIsDown(66): return 'b'
        elif keyIsDown(69): return 'e'
        else: sleep(0.02)



if __name__ == '__main__':
    visualizationPaths, dataPaths = getOutputDataPaths()

    unwrappedPaths = [p for person in visualizationPaths 
                            for lr in person 
                                for p in lr]

    chrome_path = "C:\Program Files\Google\Chrome\Application\chrome.exe"
    ratings = []
    for imagePath in unwrappedPaths:
        #imagePath = visualizationPaths[0][0][0]
        webbrowser.open_new_tab(imagePath)  
        sleep(0.5)
        #p = subprocess.Popen([visualizationPaths[0][0][0]], shell = True)
        print("Is the quality good (g), bad (b) or an edge case (e)? ")
        res = getRating()
            
        imageName = imagePath.split('\\')[-1].split('.pdf')[0]
        individualNumber = imagePath.split('\\')[-3]
        print(f"{imageName} was rated {res}")
        ratings.append([individualNumber,imageName, res])

    with open('dataRatings.txt', 'w') as f:
        for rating in ratings:
            individualNumber,imageName, res = rating
            f.write(f"{individualNumber}, {imageName}, {res}\n")
