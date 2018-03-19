# -*- coding: utf-8 -*-
"""
Spyder Editor

emailprovider class:
    logs into an email address given
    can log events into a logstring.
    keeps track of send times for emails -> if enough time has passed, send logfile to addressee
    use mimetext for mails

@author: Martin Sanner
"""

import time
import smtplib
from email.mime.text import MIMEText as text_msg

class email_bot:
    from_addr = ""
    to_addr = ""
    
    t_last_sent = 0
    msg = ""
    header = ""
    
    
    def __init__(self,from_add,to_add,server,password,user = "",min_time = 600):
        '''
            from_add: sender address for the emails
            to_add: receiver email addresses
            server: email server for smtp protocol (ip:port)
            password: Password for the user
            user: Username, if not supplied assume from_add
            min_time: Minimum time in seconds that needs to have passed since the last mail was sent.
            
        
        '''
        self.t_last_sent = 0
        self.from_addr = from_add
        self.to_addr = to_add
        if user != "":
            self.username = user
        else:
            self.username = self.from_addr
        self.password = password
        
        self.server = server

        self.topic = ""
        self.mintime_passed = min_time
        print("Email bot activated.")
    def set_topic(self,t):
        self.topic = t
   
    def append_message(self,mess, send = True):
        '''
        append message to be sent. Then tries to send the message.
        '''
        self.msg = self.msg + mess
        if send:
            self.send_email()
    def send_email(self):
        '''
        Send an email based on the last time sent and msg
        '''
        if int(time.time() - self.t_last_sent) >self.mintime_passed: 
            smtpObj = smtplib.SMTP(self.server)
            smtpObj.starttls()
            smtpObj.login(self.username,self.password)
            email = text_msg(self.msg)
            email["Subject"] = self.topic
            email["From"] = self.from_addr
            email["To"] = self.to_addr
            smtpObj.sendmail(self.from_addr, self.to_addr, email.as_string())
            print("Message sent")
            self.t_last_sent = time.time()
            smtpObj.quit()
            
            #reset mail message
            self.msg = ""
            
        else:
            print("Too soon to send more emails - trying later.")
