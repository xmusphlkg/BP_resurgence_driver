#!/usr/bin/env Rscript
# Fixed primary prediction performance and its burden-contrast boundary.
suppressPackageStartupMessages({library(dplyr);library(ggplot2);library(patchwork);library(readr);library(ragg);library(scales)})
source('script/figures/_common.R')
path <- read_csv_checked(fd_path('figure_4_representative_matched_path.csv'), required=c('date','scenario_id','expected_notifications')) |>
  mutate(date=as.Date(date), scenario=recode(scenario_id, matched_transmission_truth='Transmission-context retained',matched_reporting_truth='Ascertainment-change retained'))
country <- read_csv_checked('manuscript/tables/country_level_primary_auc.csv', required=c('iso3','auc','dataset','package','classifier')) |>
  filter(dataset=='process_anchored',classifier=='shrinkage_linear_discriminant',package %in% c('severe','notifications')) |>
  mutate(stream=factor(if_else(package=='severe','Notifications + infant severe outcomes','Notifications alone'), levels=c('Notifications alone','Notifications + infant severe outcomes')),iso3=factor(iso3,levels=rev(pilot_iso3)))
strata <- read_csv('manuscript/tables/burden_stratified_auc.csv',show_col_types=FALSE) |>
  mutate(stream=if_else(package_id=='notifications+severe','Infant severe outcomes','Simulated index'),
         burden_stratum=factor(burden_stratum,levels=c('1.0–1.5','1.5–2.0','2.0–4.0','≥4.0')))
stopifnot(nrow(country)==18L,nrow(strata)==16L,all(is.finite(strata$country_macro_auc)))
stopifnot(all(is.finite(country$auc)),all(country$auc>=0 & country$auc<=1),
          setequal(as.character(country$iso3),pilot_iso3),
          all(table(country$iso3,country$package)==1L))
write_csv(country,fd_path('figure_2_country_primary_comparison.csv'))
country_pairs <- country |> select(iso3,package,auc) |> tidyr::pivot_wider(names_from=package,values_from=auc)
stream_pal <- c('Infant severe outcomes'=clr('orange'),'Simulated index'=clr('blue'))
p_a <- ggplot(path,aes(date,expected_notifications,colour=scenario,linetype=scenario))+
 geom_line(linewidth=.65)+geom_line(data=path |> filter(scenario=='Ascertainment-change retained'),linewidth=.65)+scale_colour_manual(values=c(clr('orange'),clr('blue')))+
 scale_linetype_manual(values=c('22','solid'))+scale_x_date(date_breaks='1 year',date_labels='%Y')+
 scale_y_continuous(labels=label_number(scale_cut=cut_short_scale()))+
 labs(tag='a',x=NULL,y='Matched expected notifications',colour=NULL,linetype=NULL)+theme_nature(grid='y')+
 theme(legend.position='bottom',legend.text=element_text(size=5.6))+guides(colour=guide_legend(ncol=1))
p_b <- ggplot(country,aes(auc,iso3,colour=stream,shape=stream))+
 geom_vline(xintercept=.5,linetype='dotted',colour=clr('grey'))+
 geom_segment(data=country_pairs,aes(x=notifications,xend=severe,y=iso3,yend=iso3),inherit.aes=FALSE,colour=clr('grey'),linewidth=.45)+
 geom_point(size=1.8)+
 scale_colour_manual(values=c('Notifications alone'=clr('grey'),'Notifications + infant severe outcomes'=clr('orange')))+
 scale_shape_manual(values=c('Notifications alone'=1,'Notifications + infant severe outcomes'=16))+
 scale_x_continuous(limits=c(0,1),breaks=seq(0,1,.25))+scale_y_discrete(labels=country_abbrev)+
 labs(tag='b',x='Country-specific held-out AUC',y=NULL,colour=NULL,shape=NULL)+theme_nature(grid='x')+theme(legend.position='bottom',legend.text=element_text(size=5.6))+guides(colour=guide_legend(ncol=1),shape=guide_legend(ncol=1))
make_strata <- function(window,tag) {
 d <- strata |> filter(burden_window==window)
 labels <- d |> filter(stream=='Infant severe outcomes') |> mutate(label=paste0(burden_stratum,'\n',n_pairs,' pairs\n',n_countries,if_else(n_countries==1,' country',' countries')))
 ggplot(d,aes(burden_stratum,country_macro_auc,colour=stream,shape=stream))+
 geom_hline(yintercept=.5,linetype='dotted',colour=clr('grey'))+
 geom_errorbar(aes(ymin=country_macro_lower_95,ymax=country_macro_upper_95),width=.1,linewidth=.4,position=position_dodge(width=.3))+
 geom_point(size=1.8,position=position_dodge(width=.3))+
 scale_colour_manual(values=stream_pal)+scale_shape_manual(values=c(16,17))+
 scale_x_discrete(labels=setNames(labels$label,labels$burden_stratum))+
 scale_y_continuous(limits=c(.25,.85),breaks=seq(.3,.8,.1))+
 labs(tag=tag,x=paste0(window,' infant burden ratio (larger / smaller)'),y='Country-macro AUC',colour=NULL,shape=NULL)+
 theme_nature(grid='y')+theme(legend.position='bottom',legend.text=element_text(size=5.6),axis.text.x=element_text(size=5.5))
}
figure <- (p_a | p_b) / (make_strata('2024','c') | make_strata('2023–2025','d')) + plot_layout(heights=c(1.05,1))
stem <- 'figure_2_matched_notification_benchmark'
grDevices::cairo_pdf(fig_path(paste0(stem,'.pdf')),width=183/25.4,height=138/25.4);print(figure);dev.off()
ragg::agg_png(fig_path(paste0(stem,'.png')),width=183,height=138,units='mm',res=600,background='white');print(figure);dev.off()
