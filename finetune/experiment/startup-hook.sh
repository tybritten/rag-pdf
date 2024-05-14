pip install pachyderm-sdk
#python -m nltk.downloader stopwords
# (4.29.24) Andrew: There is a known issue where the nltk package is trying to download the 'stopwords' resource to the '/usr/local/lib/python3.10/site-packages/llama_index/core/_static/nltk_cache/corpora' directory, but it doesn't have the necessary permissions to do so.
# (4.29.24) Andrew Resolution: Update NLTK_DATA to a path in your host computer. There is a permission issues
python -m nltk.downloader -d /nvmefs1/andrew.mendez/nltk_cache all
export NLTK_DATA=/nvmefs1/andrew.mendez/nltk_cache
