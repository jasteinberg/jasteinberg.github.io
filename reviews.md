---
layout: page
title: Reviews
permalink: /reviews/
---

<p class="reviews-intro">
Living reference documents — syntheses of a literature organized around how the
results tension against each other, revised as the field moves. Not chronological;
each is updated in place.
</p>

<div class="post-list">
{% assign revs = site.reviews | sort: "updated" | reverse %}
{% for rev in revs %}
  <article class="post-entry">
    {% if rev.updated %}
      <p class="post-meta">Updated {{ rev.updated | date: "%B %Y" }}</p>
    {% endif %}
    <h2 class="post-title">
      <a href="{{ rev.url | relative_url }}">{{ rev.title }}</a>
    </h2>
    {% if rev.description %}
      <p class="post-desc">{{ rev.description }}</p>
    {% endif %}
  </article>
{% endfor %}
</div>
