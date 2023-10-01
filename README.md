# Project Title

## Description

This project was created with the initial purpose of learning to scrape 'messy' data and clean it through a pipeline of functions automatically.  Some features may not be implemented perfectly, or be missing entirely.  Clean, functional, decoupled code is the main purpose of this project, along with learning how to implement traditional relational databases and NoSQL databases.

## Features

- Scrapes all of the listings for search term
- Cleans data for analysis
- Stores data in database
- Event logging
- NLP of descriptions and amenities
- Machine learning and visualization of price influencers.

## Future Features

- Load different formats into database
- MongoDB integration for articles
- Visualize Data
- Machine Learning algorithms to find key price predictors.
- Options Run on Command Line
- Web Interface with more options

## File Descriptions

`trulia_scrape.py` - this file can be run from the command line and will automatically scrape apartment data for the Austin area and save it to a CSV file in the `daily_scrape_files` folder.

## How To Use

```bash
# Clone this repository
$ git clone https://github.com/datapointchris/etl_housing

# Go into the repository
$ cd etl_housing

# Run the app
$ python scraper.py
```

Program will begin scraping Trulia for rentals.  Currently only Austin rentals have been tested.  Other cities and search terms will be available in future versions.

Jupyter Notebooks are also included in the repo where you can run the program and change the `page_url` to scrape different cities.

## Requirements

You really should only need to install BeautifulSoup if you don't have it.  Everything else should be part of the standard library.

- Numpy
- Pandas
- Requests
- BeautifulSoup
- SQLite3

## Credits

## License

[MIT](https://tldrlegal.com/license/mit-license)
