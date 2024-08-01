from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="crowdsam",  # 项目名称
    version="0.1.0",  # 项目版本
    author="Zhi Cai",  # 作者姓名
    author_email="caizhi97@buaa.edu.cn",  # 作者邮箱
    description="This is crowd-sam, a project aims to automatic annotation with the help of foundation models",  # 简短描述
    long_description=long_description,  # 长描述（通常是README内容）
    long_description_content_type="text/markdown",  # 长描述的内容类型
    url="https://github.com/FelixCaae/CrowdSAM",  # 项目主页URL
    packages=find_packages(),  # 自动发现项目中的所有包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Python版本要求
    install_requires=requirements,  # 项目依赖项
    # entry_points={
    #     'console_scripts': [
    #         'my_project=my_module.module1:main_function',  # 命令行工具入口
    #     ],
    # },
)