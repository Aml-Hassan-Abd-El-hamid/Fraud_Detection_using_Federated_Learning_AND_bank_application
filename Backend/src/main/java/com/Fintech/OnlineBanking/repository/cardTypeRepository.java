package com.Fintech.OnlineBanking.repository;

import java.util.Collection;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.Fintech.OnlineBanking.domain.Transactions;
import com.Fintech.OnlineBanking.domain.UserAccounts;
import com.Fintech.OnlineBanking.domain.CardType;

public interface cardTypeRepository extends JpaRepository <CardType, Integer>{

	List<CardType> findAll();

}
