rm -r -f db.sqlite3
rm -r -f /data/poetry/
python3 manage.py migrate
python3 manage.py loaddata datasets/django/all_django.json
python3 manage.py createinitialrevisions
python3 manage.py generate_markups --db
python3 manage.py rebuild_index --noinput
