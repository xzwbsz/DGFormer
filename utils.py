import yaml
import sys
import os

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

# 加载配置文件
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


nodename = os.uname().nodename
file_dir = config['filepath'][nodename]


def main():

    pass

if __name__ == '__main__':
    main()
