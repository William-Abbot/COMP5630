import subprocess
import pyautogui as auto
import time

screen_size = auto.size()

x = True
counter = 0

go_pos = '(1429,133)'
note_pos = '(1935, 729)'

while(x):

    go = subprocess.Popen('C:\\Users\\wabbo\\OneDrive\\Desktop\\Go-Solver-master\\Go-Solver-0.1\\Go-Solver-0.1\\Go-Solver (computer_vs_computer).exe')
    auto.typewrite('5')
    auto.press('enter')
    auto.typewrite('5')
    auto.press('enter')
    auto.typewrite('6')
    auto.press('enter')
    time.sleep(5)
    auto.hotkey('ctrlleft','a')
    auto.hotkey('ctrlleft','c')
    
    notepad = subprocess.Popen('C:\\Windows\\System32\\notepad.exe')
    time.sleep(2)
    auto.hotkey('ctrlleft','v')
    auto.hotkey('ctrlleft','s')
    time.sleep(0.5)
    auto.typewrite('test_data_'+str(counter))
    
    notepad.kill
    go.kill
    time.sleep(2)