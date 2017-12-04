from haystack import indexes
from poetry.apps.corpus.models import Poem, Markup


class PoemIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    author = indexes.CharField(model_attr='author')

    def get_model(self):
        return Poem

    def index_queryset(self, using=None):
        """Used when the entire index for model is updated."""
        return self.get_model().objects.all()

class MarkupIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    metre = indexes.CharField(model_attr='metre')
    feet_count=indexes.CharField(model_attr='feet_count')

    def get_model(self):
        return Markup

    def index_queryset(self, using=None):
        """Used when the entire index for model is updated."""
        return self.get_model().objects.all()
