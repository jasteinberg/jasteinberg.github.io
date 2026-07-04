---
layout: page
title: Blog
permalink: /blog/
---

<div class="post-list">
{% for post in site.posts %}
  <article class="post-entry">
    <p class="post-meta">{{ post.date | date: "%B %Y" }}</p>
    <h2 class="post-title">
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </h2>
    {% if post.description %}
      <p class="post-desc">{{ post.description }}</p>
    {% endif %}
  </article>
{% endfor %}
</div>
