import logging



logging.basicConfig(level=logging.INFO,
                        filename='./logTest.log',
                        filemode='a',
                        # encoding='utf8',
                        force=True,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

logging.info("ttttestesterstestestestw")
logging.info("ttttestestersdsdsdstestestestw")
logging.info("ttttestesterssdsdstestestestw")
logging.info("ttttestestsdsdsdsw")
logging.info("ttttestestsdsdserstestestestw")
logging.info("ttttestessddsdterstestestestw")
logging.info("ttttestestedsdsdrstestestestw")
logging.info("ttttestestedsdsdsdsrstestestestw")